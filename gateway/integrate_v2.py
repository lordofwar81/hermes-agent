#!/usr/bin/env python3
"""
Integrate extracted modules into gateway/run.py.

Replaces method bodies with thin wrappers calling functions from:
- runner_init.py
- adapter_factory.py
- authorization.py
- agent_execution.py
- lifecycle.py
- message_processing.py
- voice_mode.py
- voice_reply.py
- config_loaders.py
- exit_state.py
- queue_helpers.py
"""

import re
import sys
from pathlib import Path

# Paths
RUN_PY = Path(__file__).parent.parent / "gateway" / "run.py"

# Read the file
with open(RUN_PY, encoding="utf-8") as f:
    content = f.read()
lines = content.splitlines()

# Track modifications
modifications = []

def find_method_end(start_line: int) -> int:
    """Find the end of a method definition by tracking indentation."""
    base_indent = len(lines[start_line]) - len(lines[start_line].lstrip())
    i = start_line + 1
    while i < len(lines):
        line = lines[i]
        # Empty lines don't end the method
        if not line.strip():
            i += 1
            continue
        # Dedent means method ended
        current_indent = len(line) - len(line.lstrip())
        if current_indent <= base_indent and line.strip() and not line.strip().startswith("#"):
            return i
        i += 1
    return len(lines)


def replace_method(method_name: str, new_body: list[str], start_idx: int = 0) -> bool:
    """Replace a method body with new content."""
    for i in range(start_idx, len(lines)):
        line = lines[i]
        if f"def {method_name}(" in line or f"async def {method_name}(" in line:
            # Found the method
            end_idx = find_method_end(i)
            # Keep the method signature, replace the body
            indent = len(lines[i]) - len(lines[i].lstrip())
            signature = lines[i]
            # Insert signature then new body
            lines[i:end_idx+1] = [signature] + new_body + [""]
            modifications.append(f"Replaced {method_name} (lines {i+1}-{end_idx+1})")
            return True
    return False


# ============================================================================
# 1. Add imports (add after existing gateway imports)
# ============================================================================
print("Adding imports...")

# Find the last "from gateway" import to add after
last_import_idx = -1
for i in range(len(lines)):
    if lines[i].strip().startswith("from gateway") or lines[i].strip().startswith("import gateway"):
        last_import_idx = i

if last_import_idx >= 0:
    # Check if imports already exist
    existing_imports = set()
    for i in range(max(0, last_import_idx - 20), last_import_idx + 1):
        for module in ["runner_init", "adapter_factory", "authorization", "agent_execution", "lifecycle", "message_processing", "voice_mode", "voice_reply", "config_loaders", "exit_state", "queue_helpers"]:
            if module in lines[i]:
                existing_imports.add(module)

    new_imports = []
    if "runner_init" not in existing_imports:
        new_imports.append("from gateway import runner_init")
    if "adapter_factory" not in existing_imports:
        new_imports.append("from gateway import adapter_factory")
    if "authorization" not in existing_imports:
        new_imports.append("from gateway import authorization")
    if "agent_execution" not in existing_imports:
        new_imports.append("from gateway import agent_execution")
    if "lifecycle" not in existing_imports:
        new_imports.append("from gateway import lifecycle")
    if "message_processing" not in existing_imports:
        new_imports.append("from gateway import message_processing")
    if "voice_mode" not in existing_imports:
        new_imports.append("from gateway import voice_mode")
    if "voice_reply" not in existing_imports:
        new_imports.append("from gateway import voice_reply")
    if "config_loaders" not in existing_imports:
        new_imports.append("from gateway import config_loaders")
    if "exit_state" not in existing_imports:
        new_imports.append("from gateway import exit_state")
    if "queue_helpers" not in existing_imports:
        new_imports.append("from gateway import queue_helpers")

    if new_imports:
        lines[last_import_idx+1:last_import_idx+1] = new_imports + [""]
        modifications.append(f"Added {len(new_imports)} import lines")


# ============================================================================
# 2. Replace _create_adapter with wrapper
# ============================================================================
print("Replacing _create_adapter...")
wrapper = [
    "        return adapter_factory.create_adapter(",
    "            runner=self,",
    "            platform=platform,",
    "            config=config,",
    "        )",
]
if replace_method("_create_adapter", wrapper):
    modifications.append("Replaced _create_adapter with wrapper")


# ============================================================================
# 3. Replace authorization methods
# ============================================================================
print("Replacing authorization methods...")

# _is_user_authorized
wrapper = [
    "        return authorization.is_user_authorization(",
    "            runner=self,",
    "            source=source,",
    "        )",
]
if replace_method("_is_user_authorized", wrapper):
    modifications.append("Replaced _is_user_authorized with wrapper")


# ============================================================================
# 4. Replace start method
# ============================================================================
print("Replacing start method...")
wrapper = [
    "        return await lifecycle.start_gateway_runner(",
    "            runner=self,",
    "        )",
]
if replace_method("start", wrapper):
    modifications.append("Replaced start with wrapper")


# ============================================================================
# 5. Replace stop method
# ============================================================================
print("Replacing stop method...")
wrapper = [
    "        return await lifecycle.stop_gateway_runner(",
    "            runner=self,",
    "            drain=drain,",
    "            reason=reason,",
    "        )",
]
if replace_method("stop", wrapper):
    modifications.append("Replaced stop with wrapper")


# ============================================================================
# 6. Replace agent execution methods
# ============================================================================
print("Replacing agent execution methods...")

# _run_agent
wrapper = [
    "        return await agent_execution.run_agent(",
    "            runner=self,",
    "            message=message,",
    "            context_prompt=context_prompt,",
    "            history=history,",
    "            source=source,",
    "            session_id=session_id,",
    "            session_key=session_key,",
    "            run_generation=run_generation,",
    "            _interrupt_depth=_interrupt_depth,",
    "            event_message_id=event_message_id,",
    "            channel_prompt=channel_prompt,",
    "        )",
]
if replace_method("_run_agent", wrapper):
    modifications.append("Replaced _run_agent with wrapper")

# _run_agent_via_proxy
wrapper = [
    "        return await agent_execution.run_agent_via_proxy(",
    "            runner=self,",
    "            message=message,",
    "            source=source,",
    "            session_key=session_key,",
    "            session_id=session_id,",
    "        )",
]
if replace_method("_run_agent_via_proxy", wrapper):
    modifications.append("Replaced _run_agent_via_proxy with wrapper")


# ============================================================================
# 7. Replace message processing methods
# ============================================================================
print("Replacing message processing methods...")

# _handle_message
wrapper = [
    "        return await message_processing.handle_message(",
    "            runner=self,",
    "            event=event,",
    "        )",
]
if replace_method("_handle_message", wrapper):
    modifications.append("Replaced _handle_message with wrapper")

# _handle_message_with_agent
wrapper = [
    "        return await message_processing.handle_message_with_agent(",
    "            runner=self,",
    "            event=event,",
    "            source=source,",
    "            _quick_key=_quick_key,",
    "            run_generation=run_generation,",
    "        )",
]
if replace_method("_handle_message_with_agent", wrapper):
    modifications.append("Replaced _handle_message_with_agent wrapper")

# _handle_active_session_busy_message
wrapper = [
    "        return await message_processing.handle_active_session_busy_message(",
    "            runner=self,",
    "            event=event,",
    "            session_key=session_key,",
    "        )",
]
if replace_method("_handle_active_session_busy_message", wrapper):
    modifications.append("Replaced _handle_active_session_busy_message with wrapper")

# _prepare_inbound_message_text
wrapper = [
    "        return await message_processing.prepare_inbound_message_text(",
    "            runner=self,",
    "            event=event,",
    "        )",
]
if replace_method("_prepare_inbound_message_text", wrapper):
    modifications.append("Replaced _prepare_inbound_message_text with wrapper")


# ============================================================================
# Write modified file
# ============================================================================
print(f"\nTotal modifications: {len(modifications)}")
for mod in modifications:
    print(f"  - {mod}")

print(f"\nWriting {RUN_PY}...")
with open(RUN_PY, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Original lines: {len(content.splitlines())}")
print(f"New lines: {len(lines)}")
print(f"Lines removed: {len(content.splitlines()) - len(lines)}")
