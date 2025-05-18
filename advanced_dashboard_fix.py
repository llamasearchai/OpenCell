#!/usr/bin/env python3
import re

file_path = 'src/workflow/pipeline/workflow_manager.py'

with open(file_path, 'r') as f:
    lines = f.readlines()

# Find the insertion point for logs_display_content and the line to modify
insertion_line_index = -1
log_interpolation_line_index = -1

# Marker for the start of the target f-string block (simplified)
# This assumes the f-string block for logs is the last major f-string concatenation to `html`
# within the `generate_workflow_monitor_dashboard` try block before the `return html`
# A more robust marker might be needed if the structure is very complex.

# We are looking for the pattern:
#   ... some code ...
#   # Complete the HTML
#   html += f"""  <--- target_fstring_start_index
#       ...
#       {(workflow_logs or ...).replace(...)} <--- log_interpolation_line_index
#       ...
#   """ 

# Find the line '# Complete the HTML'
complete_html_marker_index = -1
for i, line in enumerate(lines):
    if "# Complete the HTML" in line:
        complete_html_marker_index = i
        break

if complete_html_marker_index != -1:
    # Find the `html += f"""` that follows, this should be the start of the logs block
    for i in range(complete_html_marker_index + 1, len(lines)):
        # This regex is a bit broad, hoping it's unique enough in this context
        if re.search(r"html \s*\+=\s*f\"\"\"", lines[i]):
            # Check if this block contains the logs-container, a good sign it's the right one
            block_is_logs_panel = False
            for j in range(i, min(i + 20, len(lines))):
                if 'class="panel logs"' in lines[j] or 'class="logs-container"' in lines[j]:
                    block_is_logs_panel = True
                    break
            if block_is_logs_panel:
                insertion_line_index = i # Insert before this line
                # Now find the log interpolation line within this block
                for k in range(i + 1, min(i + 15, len(lines))): # Search a few lines down
                    if "(workflow_logs or" in lines[k] and ".replace(" in lines[k]:
                        log_interpolation_line_index = k
                        break
                break # Found the target f-string block

if insertion_line_index != -1 and log_interpolation_line_index != -1:
    # Define the new lines
    # Ensure these strings represent single complete lines of Python code,
    # newlines for the file will be added when writing.
    variable_definition_line1_str = "            logs_display_content = (workflow_logs or \"No logs available\").replace('\\n', '<br>')"
    variable_definition_line2_str = "            logs_display_content = logs_display_content.replace(r'\\n', '<br>') # Handle literal \\n"
    
    # Get current indentation of the log_interpolation_line
    current_indent = len(lines[log_interpolation_line_index]) - len(lines[log_interpolation_line_index].lstrip())
    new_log_line = ' ' * current_indent + '{logs_display_content}\n'

    # Perform modifications
    original_log_line_content = lines[log_interpolation_line_index].strip()
    print(f"Found log interpolation at line {log_interpolation_line_index + 1}: {original_log_line_content}")
    print(f"Will replace with: {new_log_line.strip()}")
    print(f"Will insert variable definition before line {insertion_line_index + 1}")

    # Insert the variable definition lines
    # Adjust indentation of definition lines to match the context (e.g., indentation of `html +=`) if needed.
    # Assuming the `html += f"""` is at the same indent level as where `logs_display_content` should be defined.
    indent_of_insertion_point_line = len(lines[insertion_line_index]) - len(lines[insertion_line_index].lstrip())
    
    lines.insert(insertion_line_index, ' ' * indent_of_insertion_point_line + variable_definition_line2_str.lstrip() + '\n') # insert second line first due to index shift
    lines.insert(insertion_line_index, ' ' * indent_of_insertion_point_line + variable_definition_line1_str.lstrip() + '\n')
    
    # Adjust log_interpolation_line_index due to insertions
    log_interpolation_line_index += 2 
    
    # Replace the log interpolation line
    lines[log_interpolation_line_index] = new_log_line
    
    with open(file_path, 'w') as f:
        f.writelines(lines)
    print(f"Successfully modified {file_path} for log display.")
else:
    if complete_html_marker_index == -1:
        print("Error: Could not find '# Complete the HTML' marker.")
    elif insertion_line_index == -1:
        print("Error: Could not find the target 'html += f\\\"\\\"\\\"' block for logs.")
    elif log_interpolation_line_index == -1:
        print("Error: Could not find the log interpolation line within the target block.")
    print("Script did not make changes.") 