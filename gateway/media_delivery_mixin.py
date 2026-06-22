"""Post-stream MEDIA delivery for ``GatewayRunner``.

Round 44 of the god-file decomposition. Lifted verbatim from
gateway/run.py into gateway/media_delivery_mixin.py.

``_deliver_media_from_response`` extracts MEDIA: tags and local file
paths from a response and delivers them. Called after streaming has
already sent the text to the user, so the text itself is already
delivered — this only handles file attachments that the normal
``_process_message_background`` path would have caught.

It captures ``[[as_document]]`` before ``extract_media`` strips it, so
image-extension files route through ``send_document`` (preserving bytes)
instead of ``send_multiple_images`` (Telegram sendPhoto recompresses to
~1280px). It chains the cleaned text through each extractor
(extract_media → extract_images → extract_local_files) so MEDIA: tags
and image URLs are removed before the bare-path auto-detect runs,
avoiding false-positive bare-path matches with the MEDIA: prefix glued
on (matches the chain order in gateway/platforms/base.py). It partitions
images for batched delivery, and routes non-image media by extension
(audio / video / document).

``self.*`` references resolve unchanged via the MRO. Behavior-neutral
lift matching the existing mixin pattern.

``gateway.run`` module-level runtime global (``logger``) is
lazy-imported at the top of the method body to avoid the circular import
(``gateway.run`` imports this mixin at module top). The non-circular
module symbols ``MessageEvent`` (from gateway.platforms.base) and
``_thread_metadata_for_source`` (from gateway.gateway_message_pipeline)
are imported at module top. ``Path`` and ``_quote`` (from urllib.parse)
and ``BasePlatformAdapter`` + ``should_send_media_as_audio`` (from
gateway.platforms.base) are imported in-body at the top of the method
(already lazy in source) and kept verbatim — the body's own in-body
imports shadow any module-level names, so lifting them to the mixin top
is unnecessary and would diverge from the source.
"""

from __future__ import annotations

from gateway.platforms.base import MessageEvent
from gateway.gateway_message_pipeline import _thread_metadata_for_source


class MediaDeliveryMixin:
    async def _deliver_media_from_response(
        self,
        response: str,
        event: MessageEvent,
        adapter,
    ) -> None:
        """Extract MEDIA: tags and local file paths from a response and deliver them.

        Called after streaming has already sent the text to the user, so the
        text itself is already delivered — this only handles file attachments
        that the normal _process_message_background path would have caught.
        """
        from gateway.run import logger

        from pathlib import Path
        from urllib.parse import quote as _quote

        try:
            # Capture [[as_document]] before extract_media strips it, so the
            # dispatch partition below can route image-extension files
            # through send_document (preserving bytes) instead of
            # send_multiple_images (Telegram sendPhoto recompresses to ~1280px).
            force_document_attachments = "[[as_document]]" in response

            from gateway.platforms.base import BasePlatformAdapter, should_send_media_as_audio

            media_files, cleaned = adapter.extract_media(response)
            media_files = BasePlatformAdapter.filter_media_delivery_paths(media_files)
            # Chain the cleaned text through each extractor (extract_media →
            # extract_images → extract_local_files) so MEDIA: tags and image URLs
            # are removed before the bare-path auto-detect runs. Previously the
            # cleaned text from extract_media was dropped (``_``) and
            # extract_local_files scanned text that still contained MEDIA: tags,
            # producing false-positive bare-path matches with the MEDIA: prefix
            # glued on. This matches the chain order in gateway/platforms/base.py.
            _, cleaned = adapter.extract_images(cleaned)
            local_files, _ = adapter.extract_local_files(cleaned)
            local_files = BasePlatformAdapter.filter_local_delivery_paths(local_files)

            _thread_meta = _thread_metadata_for_source(event.source, self._reply_anchor_for_event(event))

            _VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.3gp'}
            _IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}

            # Partition out images so they can be sent as a single batch
            # (e.g. Signal's multi-attachment RPC). When [[as_document]] was
            # set, image-extension files skip the photo path and route to
            # send_document below — preserving original bytes.
            image_paths: list = []
            non_image_media: list = []
            for media_path, is_voice in media_files:
                ext = Path(media_path).suffix.lower()
                if (ext in _IMAGE_EXTS
                        and not is_voice
                        and not force_document_attachments):
                    image_paths.append(media_path)
                else:
                    non_image_media.append((media_path, is_voice))

            non_image_local: list = []
            for file_path in local_files:
                if (Path(file_path).suffix.lower() in _IMAGE_EXTS
                        and not force_document_attachments):
                    image_paths.append(file_path)
                else:
                    non_image_local.append(file_path)

            if image_paths:
                try:
                    images = [(f"file://{_quote(p)}", "") for p in image_paths]
                    await adapter.send_multiple_images(
                        chat_id=event.source.chat_id,
                        images=images,
                        metadata=_thread_meta,
                    )
                except Exception as e:
                    logger.warning("[%s] Post-stream image batch delivery failed: %s", adapter.name, e)

            for media_path, is_voice in non_image_media:
                try:
                    ext = Path(media_path).suffix.lower()
                    if should_send_media_as_audio(event.source.platform, ext, is_voice=is_voice):
                        await adapter.send_voice(
                            chat_id=event.source.chat_id,
                            audio_path=media_path,
                            metadata=_thread_meta,
                        )
                    elif ext in _VIDEO_EXTS:
                        await adapter.send_video(
                            chat_id=event.source.chat_id,
                            video_path=media_path,
                            metadata=_thread_meta,
                        )
                    else:
                        await adapter.send_document(
                            chat_id=event.source.chat_id,
                            file_path=media_path,
                            metadata=_thread_meta,
                        )
                except Exception as e:
                    logger.warning("[%s] Post-stream media delivery failed: %s", adapter.name, e)

            for file_path in non_image_local:
                try:
                    ext = Path(file_path).suffix.lower()
                    if ext in _VIDEO_EXTS:
                        await adapter.send_video(
                            chat_id=event.source.chat_id,
                            video_path=file_path,
                            metadata=_thread_meta,
                        )
                    else:
                        await adapter.send_document(
                            chat_id=event.source.chat_id,
                            file_path=file_path,
                            metadata=_thread_meta,
                        )
                except Exception as e:
                    logger.warning("[%s] Post-stream file delivery failed: %s", adapter.name, e)

        except Exception as e:
            logger.warning("Post-stream media extraction failed: %s", e)
