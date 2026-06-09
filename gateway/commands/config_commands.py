"""
Config command handler mixin.

This mixin contains command handlers for configuration-related commands:
- /model - Switch model for session or globally
- /reasoning - Configure reasoning behavior
- /fast - Toggle fast mode
- /verbose - Toggle verbose mode
- /compress - Configure compression
- /yolo - Toggle auto-approval mode
- /personality - Configure personality
- /codex-runtime - Configure Codex runtime
"""

import logging
from pathlib import Path
from typing import Optional, Union

from agent.i18n import t as _t
from gateway.platforms.base import EphemeralReply, MessageEvent

logger = logging.getLogger(__name__)

_hermes_home = Path.home() / ".hermes"


class ConfigCommandMixin:
    """Configuration command handlers.

    This mixin provides handlers for commands that modify gateway or agent behavior:
    - Model switching and provider selection
    - Reasoning and thinking mode configuration
    - Performance toggles (fast, verbose, compress)
    - Safety overrides (yolo mode)
    - Personality and Codex settings

    Relies on GatewayRunner state accessed via self.
    """

    async def _handle_model_command(self, event: MessageEvent) -> Optional[str]:
        """Handle /model command — switch model for this session.

        Supports:
          /model                              — interactive picker or text list
          /model <name>                       — switch for this session only
          /model <name> --global              — switch and persist to config.yaml
          /model <name> --provider <provider> — switch provider + model
          /model --provider <provider>        — switch to provider, auto-detect model
        """
        import yaml
        from hermes_cli.model_switch import (
            switch_model as _switch_model,
            parse_model_flags,
            list_authenticated_providers,
            list_picker_providers,
        )
        from hermes_cli.providers import get_label

        raw_args = event.get_command_args().strip()

        # Parse flags
        model_input, explicit_provider, persist_global, force_refresh = parse_model_flags(raw_args)

        # --refresh: bust the disk cache
        if force_refresh:
            try:
                from hermes_cli.models import clear_provider_models_cache
                clear_provider_models_cache()
            except Exception:
                pass

        # Read current config
        current_model = ""
        current_provider = "openrouter"
        current_base_url = ""
        current_api_key = ""
        user_provs = None
        custom_provs = None
        config_path = _hermes_home / "config.yaml"

        try:
            from gateway.utils.config_resolvers import _load_gateway_config
            cfg = _load_gateway_config()
            if cfg:
                model_cfg = cfg.get("model", {})
                if isinstance(model_cfg, dict):
                    current_model = model_cfg.get("default", "")
                    current_provider = model_cfg.get("provider", current_provider)
                    current_base_url = model_cfg.get("base_url", "")
                user_provs = cfg.get("providers")
                try:
                    from hermes_cli.config import get_compatible_custom_providers
                    custom_provs = get_compatible_custom_providers(cfg)
                except Exception:
                    custom_provs = cfg.get("custom_providers")
        except Exception:
            pass

        # Check session override
        source = event.source
        session_key = self._session_key_for_source(source)
        override = self._session_model_overrides.get(session_key, {})
        if override:
            current_model = override.get("model", current_model)
            current_provider = override.get("provider", current_provider)
            current_base_url = override.get("base_url", current_base_url)
            current_api_key = override.get("api_key", current_api_key)

        # No args: show interactive picker or text list
        if not model_input and not explicit_provider:
            adapter = self.adapters.get(source.platform)
            has_picker = (
                adapter is not None
                and getattr(type(adapter), "send_model_picker", None) is not None
            )

            if has_picker:
                try:
                    providers = list_picker_providers(
                        current_provider=current_provider,
                        current_base_url=current_base_url,
                        current_model=current_model,
                        user_providers=user_provs,
                        custom_providers=custom_provs,
                        max_models=50,
                    )
                except Exception:
                    providers = []

                if providers:
                    # Build picker callback
                    _self = self
                    _session_key = session_key
                    _cur_model = current_model
                    _cur_provider = current_provider
                    _cur_base_url = current_base_url
                    _cur_api_key = current_api_key

                    async def _on_model_selected(
                        _chat_id: str, model_id: str, provider_slug: str
                    ):
                        """Callback when user picks a model from the interactive picker."""
                        try:
                            result = await _switch_model(
                                model_id,
                                provider_slug,
                                session_key=_session_key,
                                persist_global=False,
                                current_model=_cur_model,
                                current_provider=_cur_provider,
                                current_base_url=_cur_base_url,
                                current_api_key=_cur_api_key,
                                config_path=config_path,
                                user_providers=user_provs,
                                custom_provs=custom_provs,
                            )
                            # Note: Full implementation includes sending result to user
                            logger.info("Model selected via picker: %s @ %s", model_id, provider_slug)
                        except Exception as e:
                            logger.error("Model switch failed: %s", e, exc_info=True)

                    # Send the picker
                    try:
                        return await type(adapter).send_model_picker(
                            adapter,
                            chat_id=source.chat_id,
                            providers=providers,
                            on_select=_on_model_selected,
                        )
                    except Exception as e:
                        logger.error("Failed to send model picker: %s", e, exc_info=True)

            # Fall back to text list
            try:
                providers = list_authenticated_providers(
                    user_providers=user_provs,
                    custom_provs=custom_provs,
                )
            except Exception:
                providers = []

            if not providers:
                return "⚠️ No authenticated providers found. Configure API keys in config.yaml."

            lines = ["**Authenticated Providers**"]
            for prov in providers:
                prov_label = get_label(prov)
                if prov == current_provider:
                    prov_label = f"✓ {prov_label}"
                lines.append(f"  {prov_label}")

            lines.append("\n**Usage:**")
            lines.append("  /model <name> — Switch model for this session")
            lines.append("  /model <name> --global — Persist to config.yaml")
            lines.append("  /model --provider <provider> — Switch provider")

            return "\n".join(lines)

        # With args: perform the switch
        try:
            result = await _switch_model(
                model_input,
                explicit_provider,
                session_key=session_key,
                persist_global=persist_global,
                current_model=current_model,
                current_provider=current_provider,
                current_base_url=current_base_url,
                current_api_key=current_api_key,
                config_path=config_path,
                user_providers=user_provs,
                custom_provs=custom_provs,
            )
            return result
        except Exception as e:
            logger.error("Model command failed: %s", e, exc_info=True)
            return f"⚠️ Model switch failed: {e}"

    async def _handle_reasoning_command(self, event: MessageEvent) -> str:
        """Handle /reasoning command — configure reasoning behavior."""
        args = event.get_command_args().strip().lower()
        source = event.source
        session_key = self._session_key_for_source(source)

        if not args:
            # Show current reasoning setting
            override = self._session_reasoning_overrides.get(session_key)
            if override is not None:
                return f"Reasoning: {'enabled' if override else 'disabled'} (session override)"
            # Check config default
            try:
                from gateway.utils.config_resolvers import _load_gateway_config
                cfg = _load_gateway_config()
                if cfg:
                    agent_cfg = cfg.get("agent", {})
                    reasoning = agent_cfg.get("reasoning", False)
                    return f"Reasoning: {'enabled' if reasoning else 'disabled'} (default)"
            except Exception:
                pass
            return "Reasoning: disabled (default)"

        # Parse args
        if args in ("on", "true", "1", "enable", "enabled"):
            self._set_session_reasoning_override(session_key, True)
            return "✓ Reasoning enabled for this session"
        elif args in ("off", "false", "0", "disable", "disabled"):
            self._set_session_reasoning_override(session_key, False)
            return "✓ Reasoning disabled for this session"
        else:
            return "Usage: /reasoning <on|off>"

    async def _handle_fast_command(self, event: MessageEvent) -> str:
        """Handle /fast command — toggle fast mode (reduced reasoning)."""
        args = event.get_command_args().strip().lower()
        source = event.source
        session_key = self._session_key_for_source(source)

        if not args:
            # Show current fast mode setting
            override = self._session_fast_overrides.get(session_key)
            if override is not None:
                return f"Fast mode: {'enabled' if override else 'disabled'} (session override)"
            return "Fast mode: disabled (default)"

        if args in ("on", "true", "1", "enable", "enabled"):
            self._session_fast_overrides[session_key] = True
            return "✓ Fast mode enabled for this session"
        elif args in ("off", "false", "0", "disable", "disabled"):
            self._session_fast_overrides[session_key] = False
            return "✓ Fast mode disabled for this session"
        else:
            return "Usage: /fast <on|off>"

    async def _handle_verbose_command(self, event: MessageEvent) -> str:
        """Handle /verbose command — toggle verbose mode."""
        args = event.get_command_args().strip().lower()
        source = event.source
        session_key = self._session_key_for_source(source)

        if not args:
            # Show current verbose setting
            override = self._session_verbose_overrides.get(session_key)
            if override is not None:
                return f"Verbose: {'enabled' if override else 'disabled'} (session override)"
            return "Verbose: disabled (default)"

        if args in ("on", "true", "1", "enable", "enabled"):
            self._session_verbose_overrides[session_key] = True
            return "✓ Verbose mode enabled for this session"
        elif args in ("off", "false", "0", "disable", "disabled"):
            self._session_verbose_overrides[session_key] = False
            return "✓ Verbose mode disabled for this session"
        else:
            return "Usage: /verbose <on|off>"

    async def _handle_compress_command(self, event: MessageEvent) -> str:
        """Handle /compress command — configure context compression."""
        args = event.get_command_args().strip().lower()
        source = event.source
        session_key = self._session_key_for_source(source)

        if not args:
            # Show current compress setting
            override = self._session_compress_overrides.get(session_key)
            if override is not None:
                return f"Compression: {'enabled' if override else 'disabled'} (session override)"
            return "Compression: enabled (default)"

        if args in ("off", "false", "0", "disable", "disabled"):
            self._session_compress_overrides[session_key] = False
            return "✓ Compression disabled for this session"
        elif args in ("on", "true", "1", "enable", "enabled"):
            self._session_compress_overrides[session_key] = True
            return "✓ Compression enabled for this session"
        else:
            return "Usage: /compress <on|off>"

    async def _handle_yolo_command(self, event: MessageEvent) -> Union[str, EphemeralReply]:
        """Handle /yolo command — toggle auto-approval for dangerous commands."""
        args = event.get_command_args().strip().lower()
        source = event.source
        session_key = self._session_key_for_source(source)

        if not args:
            # Show current yolo state
            yolo_state = self._session_yolo_states.get(session_key, False)
            return f"Auto-approval: {'enabled' if yolo_state else 'disabled'}"

        if args in ("on", "true", "1", "enable", "enabled"):
            self._session_yolo_states[session_key] = True
            return EphemeralReply("⚠️ Auto-approval ENABLED for this session — dangerous commands will execute without confirmation")
        elif args in ("off", "false", "0", "disable", "disabled"):
            self._session_yolo_states[session_key] = False
            return "✓ Auto-approval disabled for this session"
        else:
            return "Usage: /yolo <on|off>"

    async def _handle_personality_command(self, event: MessageEvent) -> str:
        """Handle /personality command — configure agent personality."""
        args = event.get_command_args().strip()
        source = event.source
        session_key = self._session_key_for_source(source)

        if not args:
            # Show current personality
            override = self._session_personality_overrides.get(session_key)
            if override:
                return f"Personality: {override}"
            return "Personality: default"

        # Set personality for session
        self._session_personality_overrides[session_key] = args
        return f"✓ Personality set to: {args}"

    async def _handle_codex_runtime_command(self, event: MessageEvent) -> str:
        """Handle /codex-runtime command — configure Codex runtime settings."""
        args = event.get_command_args().strip().lower()
        source = event.source
        session_key = self._session_key_for_source(source)

        if not args:
            # Show current codex runtime setting
            override = self._session_codex_runtime_overrides.get(session_key)
            if override is not None:
                return f"Codex runtime: {override}"
            return "Codex runtime: default"

        if args == "local":
            self._session_codex_runtime_overrides[session_key] = "local"
            return "✓ Codex runtime set to: local"
        elif args == "remote":
            self._session_codex_runtime_overrides[session_key] = "remote"
            return "✓ Codex runtime set to: remote"
        elif args == "default":
            self._session_codex_runtime_overrides.pop(session_key, None)
            return "✓ Codex runtime reset to default"
        else:
            return "Usage: /codex-runtime <local|remote|default>"
