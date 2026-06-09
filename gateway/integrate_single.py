#!/usr/bin/env python3
"""Integrate modules one method at a time with precise line-based replacement."""

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
                if '"""' in lines[j]:
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
            return start, doc_end, next_method
    return None

# Step 1: Add import
print("Step 1: Adding import...")
for i, line in enumerate(lines):
    if "from gateway.delivery import DeliveryRouter" in line:
        if "from gateway import runner_checks" not in lines[i+1]:
            lines.insert(i+1, "from gateway import runner_checks\n")
            print(f"  Inserted import at line {i+2}")
        break

with open("gateway/run.py", 'w') as f:
    f.writelines(lines)

assert verify_syntax(), "Syntax failed after import"
print("  ✓ Syntax valid\n")

# Reload after import change
with open("gateway/run.py", 'r') as f:
    lines = f.readlines()

# Step 2: Replace _warn_if_docker_media_delivery_is_risky
print("Step 2: Replacing _warn_if_docker_media_delivery_is_risky...")
bounds = find_method_lines("def _warn_if_docker_media_delivery_is_risky(self)")
if bounds:
    start, doc_end, next_method = bounds
    print(f"  Method at {start}, doc ends {doc_end}, next at {next_method}")
    # Replace lines doc_end through next_method-1 with wrapper
    # Note: need 8 spaces total (4 for class method + 4 for function body)
    new_lines = lines[:doc_end]  # Keep through docstring end
    new_lines.append("        return runner_checks.warn_if_docker_media_delivery_is_risky(self.config)\n")
    new_lines.extend(lines[next_method-1:])  # From next method
    lines = new_lines
    with open("gateway/run.py", 'w') as f:
        f.writelines(lines)
    assert verify_syntax(), "Syntax failed"
    print(f"  ✓ Replaced ({len(lines)} lines)\n")
else:
    print("  ⚠ Method not found\n")

# Step 3: Reload and replace _has_setup_skill
with open("gateway/run.py", 'r') as f:
    lines = f.readlines()

print("Step 3: Replacing _has_setup_skill...")
bounds = find_method_lines("def _has_setup_skill(self)")
if bounds:
    start, doc_end, next_method = bounds
    print(f"  Method at {start}, doc ends {doc_end}, next at {next_method}")
    new_lines = lines[:doc_end]
    new_lines.append("        return runner_checks.has_setup_skill()\n")
    new_lines.extend(lines[next_method-1:])
    lines = new_lines
    with open("gateway/run.py", 'w') as f:
        f.writelines(lines)
    assert verify_syntax(), "Syntax failed"
    print(f"  ✓ Replaced ({len(lines)} lines)\n")
else:
    print("  ⚠ Method not found\n")

# Step 4: Reload and replace _adapter_disconnect_timeout_secs
with open("gateway/run.py", 'r') as f:
    lines = f.readlines()

print("Step 4: Replacing _adapter_disconnect_timeout_secs...")
bounds = find_method_lines("def _adapter_disconnect_timeout_secs(self)")
if bounds:
    start, doc_end, next_method = bounds
    print(f"  Method at {start}, doc ends {doc_end}, next at {next_method}")
    new_lines = lines[:doc_end]
    new_lines.append("        return runner_checks.adapter_disconnect_timeout_secs()\n")
    new_lines.extend(lines[next_method-1:])
    lines = new_lines
    with open("gateway/run.py", 'w') as f:
        f.writelines(lines)
    assert verify_syntax(), "Syntax failed"
    print(f"  ✓ Replaced ({len(lines)} lines)\n")
else:
    print("  ⚠ Method not found\n")

# Step 5: Reload and replace _platform_connect_timeout_secs
with open("gateway/run.py", 'r') as f:
    lines = f.readlines()

print("Step 5: Replacing _platform_connect_timeout_secs...")
bounds = find_method_lines("def _platform_connect_timeout_secs(self)")
if bounds:
    start, doc_end, next_method = bounds
    print(f"  Method at {start}, doc ends {doc_end}, next at {next_method}")
    new_lines = lines[:doc_end]
    new_lines.append("        return runner_checks.platform_connect_timeout_secs()\n")
    new_lines.extend(lines[next_method-1:])
    lines = new_lines
    with open("gateway/run.py", 'w') as f:
        f.writelines(lines)
    assert verify_syntax(), "Syntax failed"
    print(f"  ✓ Replaced ({len(lines)} lines)\n")
else:
    print("  ⚠ Method not found\n")

print(f"\nFinal line count: {len(lines)}")
