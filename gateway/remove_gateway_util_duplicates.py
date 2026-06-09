#!/usr/bin/env python3
"""
Remove duplicate gateway utility functions from run.py.

These functions are duplicated in gateway/utils/gateway_helpers.py.
After removing, we'll add re-exports for backward compatibility.
"""

import re

# Read run.py
with open('gateway/run.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Functions to remove (name, approximate start line to verify)
# These are the duplicate functions that exist in gateway_helpers.py
duplicates = [
    '_gateway_platform_value',
    '_is_transient_network_error',
    '_gateway_loop_exception_handler',
    '_redact_gateway_user_facing_secrets',
    '_gateway_provider_error_reply',
    '_looks_like_gateway_provider_error',
    '_sanitize_gateway_final_response',
    '_prepare_gateway_status_message',
    '_send_or_update_status_coro',
    '_telegramize_command_mentions',
    '_coerce_gateway_timestamp',
]

# Pattern to match function definitions (including async)
func_pattern = re.compile(
    r'^\s*(?:async\s+)?def\s+(' + '|'.join(duplicates) + r')\s*\(',
    re.MULTILINE
)

# Find all matches and their positions
matches = list(func_pattern.finditer(content))

if not matches:
    print("No duplicate functions found to remove")
    exit(0)

# Process in reverse order to maintain line numbers
lines_removed = 0
for match in reversed(matches):
    func_name = match.group(1)
    start_pos = match.start()

    # Find the end of the function by finding the next function definition
    # or class definition at the same or lower indentation level
    remaining = content[start_pos:]

    # Find the end by looking for the next non-indented def/class or end of file
    # Get the indentation of the current function
    current_indent_match = re.match(r'^(\s*)', content[start_pos:])
    base_indent = len(current_indent_match.group(1)) if current_indent_match else 0

    # Scan through to find the end
    pos = match.end() - start_pos  # Start after the function definition
    end_pos = pos

    # Look for the next line that's not indented more than base
    # and starts with "def", "async def", "class", or is empty
    while pos < len(remaining):
        line_match = re.match(r'^.*?$', remaining[pos:], re.MULTILINE)
        if not line_match:
            break

        line_start = pos
        line = remaining[pos:pos+line_match.end()]
        pos = line_match.end()

        # Skip empty lines
        if not line.strip():
            continue

        # Check indentation
        line_indent = len(line) - len(line.lstrip())

        # If we're back to base or lower indentation and it's a new definition
        if line_indent <= base_indent:
            if re.match(r'^\s*(?:async\s+)?def\s+|^\s*class\s+', line):
                end_pos = line_start
                break
        # Otherwise continue (we're inside the function body)

    # If we didn't find a proper end, just skip this one
    if end_pos <= match.end() - start_pos:
        print(f"Skipping {func_name} - couldn't find end")
        continue

    # Remove the function (from start to end, plus trailing empty lines)
    before = content[:start_pos]
    after = content[start_pos + end_pos:]

    # Remove trailing empty lines
    while after.startswith('\n'):
        after = after[1:]

    content = before + after
    lines_removed += 1

    print(f"Removed {func_name}")

# Add re-exports at the end of imports section
# Find where to insert (after the last "from gateway" import)
import_end = content.find('from gateway.platforms.base import')
if import_end > 0:
    # Find the end of this import block
    next_newline = content.find('\n', import_end)
    insert_pos = content.find('\n', next_newline + 1)
    if insert_pos > 0:
        # Add re-exports
        re_exports = "\n# Re-export gateway helpers for backward compatibility\n"
        for func_name in duplicates:
            re_exports += f"{func_name} = gateway_helpers.{func_name}\n"

        content = content[:insert_pos] + re_exports + content[insert_pos:]

# Write back
with open('gateway/run.py', 'w', encoding='utf-8') as f:
    f.write(content)

print(f"\nRemoved {lines_removed} duplicate functions")
