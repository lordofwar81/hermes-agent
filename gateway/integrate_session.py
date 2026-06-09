#!/usr/bin/env python3
"""Integrate session_management.py and media_delivery.py modules."""

import subprocess

# Read file
with open("gateway/run.py", 'r') as f:
    lines = f.readlines()

def verify_syntax():
    """Check if current file compiles."""
    result = subprocess.run(
        ["python3", "-m", "py_compile", "gateway/run.py"],
        capture_output=True
    )
    return result.returncode == 0

def find_method_lines(method_signature):
    """Find (start_line, doc_end_line, next_method_line) for a method.
    All 1-based line numbers."""
    for i, line in enumerate(lines):
        if method_signature in line:
            start = i + 1  # 1-based
            # Find docstring end (closing """)
            doc_end = start
            in_doc = False
            for j in range(i, min(i+100, len(lines))):
                line_content = lines[j]
                # Check for single-line docstring: both """ on same line
                if line_content.count('"""') >= 2 and '"""' in line_content and not in_doc:
                    doc_end = j + 1  # Line after the docstring line
                    break
                if '"""' in line_content:
                    if not in_doc:
                        in_doc = True
                    else:
                        doc_end = j + 1  # Line after closing """
                        break
            # Find next method at same indent (4 spaces)
            next_method = None
            for j in range(doc_end, len(lines)):
                if lines[j].startswith("    def ") and "    " not in lines[j][4:]:
                    next_method = j + 1
                    break
                elif lines[j].startswith("    @property") or lines[j].startswith("    @staticmethod"):
                    next_method = j + 1
                    break
                elif lines[j].startswith("    async def ") and "    " not in lines[j][4:10]:
                    next_method = j + 1
                    break
            return start, doc_end, next_method
    return None

# Add session_management import
print("Adding session_management import...")
for i, line in enumerate(lines):
    if "from gateway import runner_checks" in line:
        if "from gateway import session_management" not in str(lines[i:i+3]):
            lines.insert(i+1, "from gateway import session_management\n")
            print(f"  Inserted import at line {i+2}")
        break

with open("gateway/run.py", 'w') as f:
    f.writelines(lines)

assert verify_syntax(), "Syntax failed after import"
print("  ✓ Syntax valid\n")

# Reload
with open("gateway/run.py", 'r') as f:
    lines = f.readlines()

# Session management methods to replace
session_methods = [
    ("def _session_key_for_source(self, source: SessionSource) -> str:",
     "        return session_management.session_key_for_source(\n            source,\n            session_store=getattr(self, 'session_store', None),\n            config=getattr(self, 'config', None),\n        )"),
    ("def _active_profile_name(self)",
     "        return session_management.active_profile_name()"),
    ("def _read_user_config(self)",
     "        return session_management.read_user_config()"),
    ("def _set_session_env(self, context: SessionContext)",
     "        return session_management.set_session_env(context)"),
    ("def _clear_session_env(self, tokens: list)",
     "        return session_management.clear_session_env(tokens)"),
]

for method_sig, wrapper in session_methods:
    print(f"Replacing {method_sig.split('(')[0].replace('def _', '')}...")
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
        assert verify_syntax(), f"Syntax failed for {method_sig}"
        print(f"  ✓ Replaced ({len(lines)} lines)\n")
        # Reload for next method
        with open("gateway/run.py", 'r') as f:
            lines = f.readlines()
    else:
        print(f"  ⚠ Method not found\n")

# Add media_delivery import
print("Adding media_delivery import...")
with open("gateway/run.py", 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "from gateway import session_management" in line:
        if "from gateway import media_delivery" not in str(lines[i:i+3]):
            lines.insert(i+1, "from gateway import media_delivery\n")
            print(f"  Inserted import at line {i+2}")
        break

with open("gateway/run.py", 'w') as f:
    f.writelines(lines)
print("  ✓ Import added\n")

# Reload
with open("gateway/run.py", 'r') as f:
    lines = f.readlines()

# Media delivery methods
media_methods = [
    ("def _consume_pending_native_image_paths(self, session_key",
     "        return media_delivery.consume_pending_native_image_paths(self, session_key)"),
]

for method_sig, wrapper in media_methods:
    print(f"Replacing {method_sig.split('(')[0].replace('def _', '')}...")
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
        assert verify_syntax(), f"Syntax failed for {method_sig}"
        print(f"  ✓ Replaced ({len(lines)} lines)\n")
    else:
        print(f"  ⚠ Method not found\n")

print(f"\nFinal line count: {len(lines)}")
