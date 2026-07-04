"""Phase 3: secondary-profile adapter registry + same-token conflict detection."""
import pytest

from gateway.run import GatewayRunner


class _FakeAdapter:
    def __init__(self, token=None):
        self.token = token


class TestCredentialFingerprint:
    def test_none_without_token(self):
        assert GatewayRunner._adapter_credential_fingerprint(_FakeAdapter()) is None

    def test_stable_and_log_safe(self):
        a = _FakeAdapter(token="secret-bot-token")
        fp1 = GatewayRunner._adapter_credential_fingerprint(a)
        fp2 = GatewayRunner._adapter_credential_fingerprint(_FakeAdapter(token="secret-bot-token"))
        assert fp1 == fp2  # stable
        assert "secret-bot-token" not in (fp1 or "")  # never the raw token
        assert len(fp1) == 16

    def test_distinct_tokens_distinct_fp(self):
        a = GatewayRunner._adapter_credential_fingerprint(_FakeAdapter(token="tok-A"))
        b = GatewayRunner._adapter_credential_fingerprint(_FakeAdapter(token="tok-B"))
        assert a != b

    def test_reads_alt_attrs(self):
        class _AltAdapter:
            def __init__(self):
                self.bot_token = "alt-token"
        assert GatewayRunner._adapter_credential_fingerprint(_AltAdapter()) is not None


class TestProfileMessageHandler:
    @pytest.mark.asyncio
    async def test_stamps_profile_on_unstamped_source(self):
        runner = GatewayRunner.__new__(GatewayRunner)
        seen = {}

        async def _fake_handle(event):
            seen["profile"] = event.source.profile
            return "ok"

        runner._handle_message = _fake_handle
        handler = runner._make_profile_message_handler("coder")

        class _Src:
            profile = None

        class _Evt:
            source = _Src()

        result = await handler(_Evt())
        assert result == "ok"
        assert seen["profile"] == "coder"

    @pytest.mark.asyncio
    async def test_does_not_override_existing_profile(self):
        runner = GatewayRunner.__new__(GatewayRunner)
        seen = {}

        async def _fake_handle(event):
            seen["profile"] = event.source.profile
            return "ok"

        runner._handle_message = _fake_handle
        handler = runner._make_profile_message_handler("coder")

        class _Src:
            profile = "writer"  # already stamped (e.g. by URL prefix)

        class _Evt:
            source = _Src()

        await handler(_Evt())
        assert seen["profile"] == "writer"


class TestPortBindingHardError:
    """A secondary profile enabling a port-binding platform aborts startup."""

    @pytest.mark.asyncio
    async def test_secondary_webhook_raises(self, monkeypatch):
        from gateway.run import MultiplexConfigError
        from gateway.config import GatewayConfig, Platform, PlatformConfig

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner._profile_adapters = {}

        # reviewer profile config enables webhook (a port-binding platform)
        reviewer_cfg = GatewayConfig(multiplex_profiles=True)
        reviewer_cfg.platforms = {
            Platform.WEBHOOK: PlatformConfig(enabled=True, extra={"port": 8644}),
        }
        monkeypatch.setattr(
            "gateway.config.load_gateway_config", lambda: reviewer_cfg
        )

        with pytest.raises(MultiplexConfigError) as ei:
            await runner._start_one_profile_adapters("reviewer", "/tmp/x", {})
        assert "webhook" in str(ei.value)
        assert "reviewer" in str(ei.value)

    @pytest.mark.asyncio
    async def test_secondary_non_binding_platform_ok(self, monkeypatch):
        """A non-port-binding platform (e.g. telegram) is NOT rejected."""
        from gateway.config import GatewayConfig, Platform, PlatformConfig

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner._profile_adapters = {}

        reviewer_cfg = GatewayConfig(multiplex_profiles=True)
        reviewer_cfg.platforms = {
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="t"),
        }
        monkeypatch.setattr(
            "gateway.config.load_gateway_config", lambda: reviewer_cfg
        )
        # _create_adapter returns None here (no real telegram token wiring), so
        # the loop simply connects nothing — the key assertion is NO raise.
        monkeypatch.setattr(runner, "_create_adapter", lambda p, c: None)

        connected = await runner._start_one_profile_adapters("reviewer", "/tmp/x", {})
        assert connected == 0  # nothing connected, but no MultiplexConfigError

    def test_port_binding_set_covers_known_listeners(self):
        from gateway.run import _PORT_BINDING_PLATFORM_VALUES
        # Every adapter that binds a TCP port must be in the guard set.
        for p in ("webhook", "api_server", "msgraph_webhook", "feishu",
                  "wecom_callback", "bluebubbles", "sms"):
            assert p in _PORT_BINDING_PLATFORM_VALUES


class _RecordingAdapter:
    """Fake adapter that records every wiring call + claims to connect cleanly."""

    def __init__(self, token="adapter-token"):
        self.token = token
        self.connected = False
        self.message_handler = None
        self.fatal_handler = None
        self.session_store = None
        self.busy_handler = None
        self.topic_recovery = None
        self.disconnected = False

    def set_message_handler(self, h):
        self.message_handler = h

    def set_fatal_error_handler(self, h):
        self.fatal_handler = h

    def set_session_store(self, s):
        self.session_store = s

    def set_busy_session_handler(self, h):
        self.busy_handler = h

    def set_topic_recovery_fn(self, fn):
        self.topic_recovery = fn


class TestStartOneProfileAdaptersHappyPath:
    """Drive _start_one_profile_adapters end-to-end so the orchestration logic
    (profile_map population, adapter wiring, connect path) is actually
    exercised. The pre-existing tests stub _create_adapter to None, which
    skips every line past the port-binding guard — leaving the connect path,
    profile_map writes, and handler wiring untested (mutation score 0%)."""

    @pytest.mark.asyncio
    async def test_successful_connect_stores_adapter_and_wires_handlers(self, monkeypatch):
        from gateway.config import GatewayConfig, Platform, PlatformConfig

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner._profile_adapters = {}
        runner.session_store = object()
        runner._busy_text_mode = "off"
        runner._handle_adapter_fatal_error = lambda *a, **k: None
        runner._handle_active_session_busy_message = lambda *a, **k: None
        runner._recover_telegram_topic_thread_id = lambda *a, **k: None

        adapter = _RecordingAdapter(token="unique-token")
        monkeypatch.setattr(runner, "_create_adapter", lambda p, c: adapter)
        # _connect_adapter_with_timeout is a staticmethod on GatewayRunner;
        # make it report success without touching the network.
        monkeypatch.setattr(
            type(runner),
            "_connect_adapter_with_timeout",
            staticmethod(lambda a, p: _async_true()),
        )
        monkeypatch.setattr(
            type(runner),
            "_safe_adapter_disconnect",
            staticmethod(lambda a, p: _async_none()),
        )

        reviewer_cfg = GatewayConfig(multiplex_profiles=True)
        reviewer_cfg.platforms = {
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="t"),
        }
        monkeypatch.setattr("gateway.config.load_gateway_config", lambda: reviewer_cfg)

        connected = await runner._start_one_profile_adapters(
            "reviewer", "/tmp/x", {}
        )

        # Happy path: adapter connected, counted, stored under the profile map.
        assert connected == 1
        assert runner._profile_adapters["reviewer"][Platform.TELEGRAM] is adapter
        # The wiring calls must have happened — these are what the profile_map
        # mutation + the message-handler stamping rely on.
        assert adapter.message_handler is not None
        assert adapter.session_store is runner.session_store
        assert adapter.fatal_handler is not None


class TestSameTokenConflictRejection:
    """Two profiles claiming the same bot credential: the duplicate must be
    detected via _adapter_credential_fingerprint and disconnected, not stored."""

    @pytest.mark.asyncio
    async def test_duplicate_token_is_rejected_and_disconnected(self, monkeypatch):
        from gateway.config import GatewayConfig, Platform, PlatformConfig

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner._profile_adapters = {}
        runner.session_store = object()
        runner._busy_text_mode = "off"
        runner._handle_adapter_fatal_error = lambda *a, **k: None
        runner._handle_active_session_busy_message = lambda *a, **k: None
        runner._recover_telegram_topic_thread_id = lambda *a, **k: None

        dup_adapter = _RecordingAdapter(token="shared-bot-token")
        monkeypatch.setattr(runner, "_create_adapter", lambda p, c: dup_adapter)
        monkeypatch.setattr(
            type(runner),
            "_connect_adapter_with_timeout",
            staticmethod(lambda a, p: _async_true()),
        )
        disconnected = []
        monkeypatch.setattr(
            type(runner),
            "_safe_adapter_disconnect",
            staticmethod(lambda a, p: _async_record(disconnected, a)),
        )

        reviewer_cfg = GatewayConfig(multiplex_profiles=True)
        reviewer_cfg.platforms = {
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="t"),
        }
        monkeypatch.setattr("gateway.config.load_gateway_config", lambda: reviewer_cfg)

        # Pre-seed `claimed` so the active profile already owns this token:
        # the reviewer's adapter (same fingerprint) must be rejected.
        active_fp = type(runner)._adapter_credential_fingerprint(
            _RecordingAdapter(token="shared-bot-token")
        )
        claimed = {(Platform.TELEGRAM, active_fp): "default"}

        connected = await runner._start_one_profile_adapters(
            "reviewer", "/tmp/x", claimed
        )

        # Not connected, not stored — conflict path taken.
        assert connected == 0
        assert Platform.TELEGRAM not in runner._profile_adapters.get("reviewer", {})
        # The duplicate was explicitly disconnected (not silently dropped).
        assert dup_adapter in disconnected


# ── async helpers for the staticmethod monkeypatches above ──────────────────

async def _async_true():
    return True


async def _async_none():
    return None


async def _async_record(record, value):
    record.append(value)
    return None


