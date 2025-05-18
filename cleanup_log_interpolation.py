#!/usr/bin/env python3

file_path = 'src/workflow/pipeline/workflow_manager.py'

with open(file_path, 'r') as f:
    lines = f.readlines()

new_lines = []
modified = False
skip_next_line = False

for i, line in enumerate(lines):
    if skip_next_line:
        skip_next_line = False
        modified = True # Mark as modified because we are deleting a line
        print(f"Deleted line {i+1}: {line.rstrip()}")
        continue

    if "{logs_display_content}" in line and '", "<br>")}' in line: # Junk is on the same line
        indent = line[:len(line) - len(line.lstrip())]
        new_line = indent + "{logs_display_content}\n"
        new_lines.append(new_line)
        print(f"Corrected line {i+1} (junk was on same line): {line.rstrip()} -> {new_line.rstrip()}")
        modified = True
    elif "{logs_display_content}" in line: # Correct case or junk is on next line
        # Ensure the line is clean if it's the target line
        if line.strip() != "{logs_display_content}":
            # It might have other things, or already be clean. 
            # If it just has {logs_display_content} and whitespace, it's fine.
            # If it has junk AFTER it on the same line, it's handled by the first condition.
            # This condition focuses on cleaning the line to ONLY have {logs_display_content}
            # if it's supposed to be that line but has other leading/trailing parts NOT caught by first condition.
            # For now, we assume if it contains {logs_display_content} it should be *just* that + indent/newline.
            # A more complex regex might be needed if this is too broad.
            indent = line[:len(line) - len(line.lstrip())]
            if line.strip() != "{logs_display_content}": # If it's not ALREADY clean
                 # Check if the *next* line has the junk, indicating this line is the one we want to make clean.
                 if i + 1 < len(lines) and '", "<br>")}' in lines[i+1].strip():
                    new_line = indent + "{logs_display_content}\n"
                    print(f"Cleaned line {i+1}: {line.rstrip()} -> {new_line.rstrip()}")
                    new_lines.append(new_line)
                    skip_next_line = True # And we will skip the junk line next iteration
                    modified = True
                 else:
                    new_lines.append(line) # Not the specific line or already clean
            else:
                new_lines.append(line) # Already clean
        else:
            new_lines.append(line)
    else:
        new_lines.append(line)

if modified:
    with open(file_path, 'w') as f:
        f.writelines(new_lines)
    print(f"File {file_path} potentially modified to clean up log interpolation.")
else:
    print(f"Log interpolation area seemed okay or pattern not found in {file_path}.") 