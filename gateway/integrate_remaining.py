#!/usr/bin/env python3
"""Integrate remaining methods from already-created modules."""

import subprocess

# Read file
with open("gateway/run.py", 'r') as f:
    lines = f.readlines()

def verify_syntax():
    result = subprocess.run(
        ["python3", "-m", "py_compile", "gateway/run.py"],
        capture_output=True
    )
    return result.returncode == 0

def find_method_lines(method_signature):
    """Find (start_line, doc_end_line, next_method_line) for a method."""
    for i, line in enumerate(lines):
        if method_signature in line:
            start = i + 1
            # Find docstring end
            doc_end = start
            in_doc = False
            for j in range(i, min(i+100, len(lines))):
                line_content = lines[j]
                if line_content.count('"""') >= 2 and '"""' in line_content and not in_doc:
                    doc_end = j + 1
                    break
                if '"""' in line_content:
                    if not in_doc:
                        in_doc = True
                    else:
                        doc_end = j + 1
                        break
            # Find next method at same indent
            next_method = None
            for j in range(doc_end, len(lines)):
                if lines[j].startswith("    def ") and "    " not in lines[j][4:]:
                    next_method = j + 1
                    break
                elif lines[j].startswith("    async def ") and "    " not in lines[j][4:10]:
                    next_method = j + 1
                    break
                elif lines[j].startswith("    @"):
                    next_method = j + 1
                    break
            return start, doc_end, next_method
    return None

# Methods to integrate
remaining_methods = [
    ("def _cache_session_source(self, session_key: str, source) -> None:",
     "        return session_management.cache_session_source(\n            session_key,\n            source,\n            getattr(self, '_session_sources', None),\n            getattr(self, '_session_sources_max', 512),\n        )"),
    ("def _get_cached_session_source(self, session_key: str):",
     "        return session_management.get_cached_session_source(\n            session_key,\n            getattr(self, '_session_sources', None),\n        )"),
    ("def _format_session_info(self)",
     "        return session_management.format_session_info(\n            resolve_gateway_model=_resolve_gateway_model,\n            load_gateway_config=_load_gateway_config,\n            resolve_runtime_agent_kwargs=_resolve_runtime_agent_kwargs,\n        )"),
    ("async def _deliver_media_from_response(self,",
     "        return await media_delivery.deliver_media_from_response(\n            response,\n            event,\n            adapter,\n            self._thread_metadata_for_source,\n            self._reply_anchor_for_event,\n        )"),
]

print(f"Starting with {len(lines)} lines\n")

for method_sig, wrapper in remaining_methods:
    method_name = method_sig.split("(")[0].replace("def _", "").replace("async def _", "")
    print(f"Replacing {method_name}...")
    bounds = find_method_lines(method_sig)
    if bounds:
        start, doc_end, next_method = bounds
        print(f"  Method at {start}, doc ends {doc_end}, next at {next_method}")
        new_lines = lines[:doc_end]
        for line in wrapper.split('\n'):
            new_lines.append(line + '\n')
        new_lines.extend(lines[next_method-1:])
        lines = new_lines
        with open("gateway/run.py", 'w') as f:
            f.writelines(lines)
        if not verify_syntax():
            print(f"  ✗ Syntax failed!")
            # Restore
            with open("gateway/run.py", 'r') as f:
                lines = f.readlines()
        else:
            print(f"  ✓ Replaced ({len(lines)} lines)\n")
        # Reload for next method
        with open("gateway/run.py", 'r') as f:
            lines = f.readlines()
    else:
        print(f"  ⚠ Method not found\n")

print(f"\nFinal line count: {len(lines)}")

# Handle _collect_auto_append_media_tags separately - it's a module-level function
print("\nChecking _collect_auto_append_media_tags (module-level)...")
with open("gateway/run.py", 'r') as f:
    content = f.read()

if "def _collect_auto_append_media_tags" in content:
    # Find and replace this function
    lines = content.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if "def _collect_auto_append_media_tags" in line:
            print(f"  Found at line {i+1}")
            # Find the end of the function
            indent = len(line) - len(line.lstrip())
            func_end = i + 1
            for j in range(i+1, len(lines)):
                # Check if we're back at module level (0 or 4 spaces indent)
                curr_indent = len(lines[j]) - len(lines[j].lstrip())
                if lines[j].strip() and curr_indent <= indent and not lines[j].strip().startswith("#"):
                    func_end = j
                    break
            print(f"  Function ends at line {func_end}")
            # Replace with import
            new_lines = lines[:i]
            new_lines.append("from gateway.media_delivery import collect_auto_append_media_tags as _collect_auto_append_media_tags\n")
            new_lines.extend(lines[func_end:])
            with open("gateway/run.py", 'w') as f:
                f.writelines(new_lines)
            print(f"  ✓ Replaced with import")
            break

print(f"\nFinal line count: {len(lines) if 'lines' in locals() else len(open('gateway/run.py').readlines())}")
