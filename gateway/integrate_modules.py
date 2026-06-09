#!/usr/bin/env python3
"""Integration script for extracted gateway modules.

Integrates:
- runner_checks.py
- session_management.py
- media_delivery.py
- runner_init.py
"""

import re
import sys

def add_imports(content: str) -> str:
    """Add imports for extracted modules."""
    # Find the import section
    import_marker = "from gateway.config import ("
    if "from gateway import runner_checks" in content:
        print("✓ runner_checks already imported")
    else:
        # Add after the gateway.config import block
        content = content.replace(
            "from gateway.delivery import DeliveryRouter",
            "from gateway.delivery import DeliveryRouter\nfrom gateway import runner_checks, session_management, media_delivery, runner_init"
        )
    return content

def integrate_warn_if_docker(content: str) -> str:
    """Replace _warn_if_docker_media_delivery_is_risky with thin wrapper."""
    # Find the method and replace body
    pattern = r'(    def _warn_if_docker_media_delivery_is_risky\(self\) -> None:.*?"""\n)(.*?)(\n    def |\n    @|\n\nclass |\Z)'
    replacement = r'''\1    return runner_checks.warn_if_docker_media_delivery_is_risky(self.config)\3'''

    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    if new_content != content:
        print("✓ _warn_if_docker_media_delivery_is_risky integrated")
    return new_content

def integrate_has_setup_skill(content: str) -> str:
    """Replace _has_setup_skill with thin wrapper."""
    pattern = r'(    def _has_setup_skill\(self\) -> bool:.*?"""\n)(.*?)(\n    # -- Voice)'
    replacement = r'''\1    return runner_checks.has_setup_skill()\3'''

    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    if new_content != content:
        print("✓ _has_setup_skill integrated")
    return new_content

def integrate_adapter_disconnect_timeout(content: str) -> str:
    """Replace _adapter_disconnect_timeout_secs with thin wrapper."""
    pattern = r'(    def _adapter_disconnect_timeout_secs\(self\) -> float:.*?"""\n)(.*?)(\n\n    def _platform_connect_timeout_secs)'
    replacement = r'''\1    return runner_checks.adapter_disconnect_timeout_secs()\3'''

    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    if new_content != content:
        print("✓ _adapter_disconnect_timeout_secs integrated")
    return new_content

def integrate_platform_connect_timeout(content: str) -> str:
    """Replace _platform_connect_timeout_secs with thin wrapper."""
    pattern = r'(    def _platform_connect_timeout_secs\(self\) -> float:.*?"""\n)(.*?)(\n\n    async def _connect_adapter_with_timeout)'
    replacement = r'''\1    return runner_checks.platform_connect_timeout_secs()\3'''

    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    if new_content != content:
        print("✓ _platform_connect_timeout_secs integrated")
    return new_content

def integrate_session_key_for_source(content: str) -> str:
    """Replace _session_key_for_source with thin wrapper."""
    pattern = r'(    def _session_key_for_source\(self, source: SessionSource\) -> str:.*?"""\n)(.*?)(\n    def _telegram_topic_mode_enabled)'
    replacement = r'''\1    return session_management.session_key_for_source(\n            source,\n            session_store=getattr(self, "session_store", None),\n            config=getattr(self, "config", None),\n        )\3'''

    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    if new_content != content:
        print("✓ _session_key_for_source integrated")
    return new_content

def integrate_cache_session_source(content: str) -> str:
    """Replace _cache_session_source with thin wrapper."""
    pattern = r'(    def _cache_session_source\(self, session_key: str, source\) -> None:\n)(.*?)(\n    def _get_cached_session_source)'
    replacement = r'''\1    return session_management.cache_session_source(\n            session_key,\n            source,\n            getattr(self, "_session_sources", None),\n            getattr(self, "_session_sources_max", 512),\n        )\3'''

    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    if new_content != content:
        print("✓ _cache_session_source integrated")
    return new_content

def integrate_get_cached_session_source(content: str) -> str:
    """Replace _get_cached_session_source with thin wrapper."""
    pattern = r'(    def _get_cached_session_source\(self, session_key: str\):)(\n)(.*?)(\n    async def _handle_message_with_agent)'
    replacement = r'''\1\2    return session_management.get_cached_session_source(\n            session_key,\n            getattr(self, "_session_sources", None),\n        )\4'''

    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    if new_content != content:
        print("✓ _get_cached_session_source integrated")
    return new_content

def integrate_active_profile_name(content: str) -> str:
    """Replace _active_profile_name with thin wrapper."""
    pattern = r'(    def _active_profile_name\(self\) -> str:.*?"""\n)(.*?)(\n    async def _kanban_notifier_watcher)'
    replacement = r'''\1    return session_management.active_profile_name()\3'''

    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    if new_content != content:
        print("✓ _active_profile_name integrated")
    return new_content

def integrate_read_user_config(content: str) -> str:
    """Replace _read_user_config with thin wrapper."""
    # Find the method at line 14904
    pattern = r'(    def _read_user_config\(self\) -> Dict\[str, Any\]:.*?"""\n)(.*?)(\n    def _set_session_env)'
    replacement = r'''\1    return session_management.read_user_config()\3'''

    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    if new_content != content:
        print("✓ _read_user_config integrated")
    return new_content

def integrate_set_session_env(content: str) -> str:
    """Replace _set_session_env with thin wrapper."""
    pattern = r'(    def _set_session_env\(self, context: SessionContext\) -> list:)(\n)(.*?)(\n    def _clear_session_env)'
    replacement = r'''\1\2    return session_management.set_session_env(context)\4'''

    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    if new_content != content:
        print("✓ _set_session_env integrated")
    return new_content

def integrate_clear_session_env(content: str) -> str:
    """Replace _clear_session_env with thin wrapper."""
    pattern = r'(    def _clear_session_env\(self, tokens: list\) -> None:)(\n)(.*?)(\n    def _resolve_session_agent_runtime)'
    replacement = r'''\1\2    return session_management.clear_session_env(tokens)\4'''

    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    if new_content != content:
        print("✓ _clear_session_env integrated")
    return new_content

def integrate_format_session_info(content: str) -> str:
    """Replace _format_session_info with thin wrapper."""
    pattern = r'(    def _format_session_info\(self\) -> str:.*?"""\n)(.*?)(\n    def _get_issue_stat)'
    replacement = r'''\1    return session_management.format_session_info(\n            resolve_gateway_model=_resolve_gateway_model,\n            load_gateway_config=_load_gateway_config,\n            resolve_runtime_agent_kwargs=_resolve_runtime_agent_kwargs,\n        )\3'''

    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    if new_content != content:
        print("✓ _format_session_info integrated")
    return new_content

def integrate_consume_pending_native_image_paths(content: str) -> str:
    """Replace _consume_pending_native_image_paths with thin wrapper."""
    pattern = r'(    def _consume_pending_native_image_paths\(self, session_key: str\) -> List\[str\]:)(\n)(.*?)(\n    def _cache_session_source)'
    replacement = r'''\1\2    return media_delivery.consume_pending_native_image_paths(self, session_key)\4'''

    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    if new_content != content:
        print("✓ _consume_pending_native_image_paths integrated")
    return new_content

def main():
    input_file = "/home/lordofwarai/.hermes/hermes-agent/gateway/run.py"

    print(f"Reading {input_file}...")
    with open(input_file, 'r') as f:
        content = f.read()

    original = content
    original_lines = len(content.splitlines())

    print("\n=== Integrating extracted modules ===\n")

    # Add imports
    content = add_imports(content)

    # Integrate runner_checks functions
    print("\n--- runner_checks.py ---")
    content = integrate_warn_if_docker(content)
    content = integrate_has_setup_skill(content)
    content = integrate_adapter_disconnect_timeout(content)
    content = integrate_platform_connect_timeout(content)

    # Integrate session_management functions
    print("\n--- session_management.py ---")
    content = integrate_session_key_for_source(content)
    content = integrate_cache_session_source(content)
    content = integrate_get_cached_session_source(content)
    content = integrate_active_profile_name(content)
    content = integrate_read_user_config(content)
    content = integrate_set_session_env(content)
    content = integrate_clear_session_env(content)
    content = integrate_format_session_info(content)

    # Integrate media_delivery functions
    print("\n--- media_delivery.py ---")
    content = integrate_consume_pending_native_image_paths(content)

    new_lines = len(content.splitlines())
    lines_removed = original_lines - new_lines

    print(f"\n=== Summary ===")
    print(f"Original lines: {original_lines}")
    print(f"New lines: {new_lines}")
    print(f"Lines removed: {lines_removed}")

    if content == original:
        print("\n⚠ No changes made - modules may already be integrated")
        return 0

    print(f"\nWriting updated file...")
    with open(input_file, 'w') as f:
        f.write(content)

    print("Validating syntax...")
    import subprocess
    result = subprocess.run(
        ["python3", "-m", "py_compile", input_file],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("✓ Syntax validation passed")
        return 0
    else:
        print("✗ Syntax validation failed:")
        print(result.stderr)
        # Restore original
        with open(input_file, 'w') as f:
            f.write(original)
        print("Restored original file")
        return 1

if __name__ == "__main__":
    sys.exit(main())
