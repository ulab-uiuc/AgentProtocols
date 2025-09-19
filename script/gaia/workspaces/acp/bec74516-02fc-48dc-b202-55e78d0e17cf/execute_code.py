
import sys
import traceback
import os
from io import StringIO

# Store original stdout for final output
original_stdout = sys.stdout

# Capture stdout for user code
captured_output = StringIO()

# Add dataset directory to sys.path if available
if os.path.exists('/dataset'):
    sys.path.insert(0, '/dataset')
    os.environ['DATASET_DIR'] = '/dataset'

# Add workspace directory to sys.path
if os.path.exists('/workspace'):
    sys.path.insert(0, '/workspace')
    os.environ['WORKSPACE_DIR'] = '/workspace'
    # Set working directory to workspace
    os.chdir('/workspace')

# Import file path resolver functionality
def resolve_dataset_file(filename):
    """Helper function to resolve dataset file paths."""
    if os.path.exists('/dataset'):
        candidate = os.path.join('/dataset', filename)
        if os.path.exists(candidate):
            return candidate
        # Try case-insensitive search
        for f in os.listdir('/dataset'):
            if f.lower() == filename.lower():
                return os.path.join('/dataset', f)
    return filename

# Make resolve_dataset_file available globally (with error handling)
try:
    # Try different ways to make function globally available
    if isinstance(__builtins__, dict):
        __builtins__['resolve_dataset_file'] = resolve_dataset_file
    elif hasattr(__builtins__, '__dict__'):
        __builtins__.__dict__['resolve_dataset_file'] = resolve_dataset_file
    else:
        # __builtins__ is a module in some environments
        setattr(__builtins__, 'resolve_dataset_file', resolve_dataset_file)
except Exception as builtins_error:
    # Fallback: add to globals and print warning
    globals()['resolve_dataset_file'] = resolve_dataset_file

try:
    # Redirect stdout to capture user output
    sys.stdout = captured_output
    
    # Print environment info to captured output
    if os.path.exists('/dataset'):
        print("Dataset directory available at /dataset")
    if os.path.exists('/workspace'):
        print("Workspace directory available at /workspace")
    
    # Create namespace for execution
    exec_globals = globals().copy()
    exec_locals = {}
    
    # Split code into lines and execute
    code_lines = '''
# Let's extract the ORCID IDs from the JSON-LD file to start the process.
import json

# Path to the JSON-LD file containing ORCID IDs
data_file_path = 'bec74516-02fc-48dc-b202-55e78d0e17cf.jsonld'

# Load the JSON-LD data
def load_jsonld(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Extract ORCID IDs
def extract_orcid_ids(data):
    ids = []
    if data.get('author') and '@id' in data['author']:
        ids.append(data['author']['@id'])
    if data.get('editor'):
        for editor in data['editor']:
            if '@id' in editor:
                ids.append(editor['@id'])
    return ids

# Main logic
json_data = load_jsonld(data_file_path)
orcid_ids = extract_orcid_ids(json_data)
orcid_ids
'''.strip().split('\n')
    
    # Execute code line by line, capturing and displaying results of expressions
    for line in code_lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        try:
            # Try to execute as statement first
            exec(line, exec_globals, exec_locals)
        except SyntaxError:
            # If it fails as statement, try as expression and print result
            try:
                result = eval(line, exec_globals, exec_locals)
                if result is not None:
                    print(result)
            except:
                # If both fail, execute as statement (this will raise the original error)
                exec(line, exec_globals, exec_locals)
    
    # Restore stdout and get captured output
    sys.stdout = original_stdout
    user_output = captured_output.getvalue()
    
    # Print success markers and output
    print("EXECUTION_SUCCESS")
    print("OUTPUT_START")
    if user_output.strip():
        print(user_output.strip())
    else:
        print("(No output)")
    print("OUTPUT_END")
    
except Exception as e:
    # Restore stdout first
    sys.stdout = original_stdout
    print("EXECUTION_ERROR") 
    print("ERROR_START")
    print(f"{type(e).__name__}: {str(e)}")
    print("TRACEBACK_START")
    traceback.print_exc()
    print("TRACEBACK_END")
    print("ERROR_END")
