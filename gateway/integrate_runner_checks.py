#!/usr/bin/env python3
"""Carefully integrate runner_checks.py module."""

import re

def main():
    input_file = "/home/lordofwarai/.hermes/hermes-agent/gateway/run.py"

    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Step 1: Add import after gateway.delivery
    new_lines = []
    import_added = False
    for i, line in enumerate(lines):
        new_lines.append(line)
        if not import_added and line.strip() == "from gateway.delivery import DeliveryRouter":
            new_lines.append("from gateway import runner_checks\n")
            import_added = True
            print(f"✓ Added import after line {i+1}")

    # Step 2: Replace _warn_if_docker_media_delivery_is_risky method
    # Find method start and end
    in_method = False
    method_start = -1
    indent = ""
    for i, line in enumerate(new_lines):
        if "    def _warn_if_docker_media_delivery_is_risky(self) -> None:" in line:
            in_method = True
            method_start = i
            indent = "    "
            continue
        if in_method:
            # Check if we've reached the next method
            if line.startswith("    def ") and "_warn_if_docker_media_delivery_is_risky" not in line:
                # Replace the method body (lines between method_start+1 and i-1)
                # Keep the docstring if present
                docstring_end = method_start + 1
                for j in range(method_start + 1, i):
                    if '"""' in new_lines[j]:
                        docstring_end = j + 1
                        break

                new_method_lines = new_lines[:method_start+1]
                # Add docstring if exists
                if docstring_end > method_start + 1:
                    new_method_lines.extend(new_lines[method_start+1:docstring_end])
                # Add the thin wrapper
                new_method_lines.append(f"{indent}return runner_checks.warn_if_docker_media_delivery_is_risky(self.config)\n")
                # Add rest of file
                new_method_lines.extend(new_lines[i:])
                new_lines = new_method_lines
                print(f"✓ Replaced _warn_if_docker_media_delivery_is_risky (lines {method_start+1}-{i})")
                break

    with open(input_file, 'w') as f:
        f.writelines(new_lines)

    print(f"Written {len(new_lines)} lines")

if __name__ == "__main__":
    main()
