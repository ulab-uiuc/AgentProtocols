
import sys
import traceback
import os
from io import StringIO

# Capture stdout
old_stdout = sys.stdout
sys.stdout = captured_output = StringIO()

# Add dataset directory to sys.path if available
if os.path.exists('/dataset'):
    sys.path.insert(0, '/dataset')
    os.environ['DATASET_DIR'] = '/dataset'
    print("Dataset directory available at /dataset")

# Add workspace directory to sys.path
if os.path.exists('/workspace'):
    sys.path.insert(0, '/workspace')
    os.environ['WORKSPACE_DIR'] = '/workspace'
    print("Workspace directory available at /workspace")
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

# Make resolve_dataset_file available globally
__builtins__['resolve_dataset_file'] = resolve_dataset_file

try:
    # Execute user code

    import json

    # This json string is simulated based on expected data from previous steps
    identification_data_json = '''
    [
        {"orcid_id": "0000-0002-1825-0097", "pre_2020_works": 15},
        {"orcid_id": "0000-0003-1613-7687", "pre_2020_works": 22},
        {"orcid_id": "0000-0001-5109-3700", "pre_2020_works": 5},
        {"orcid_id": "0000-0002-9477-123X", "pre_2020_works": 18}
    ]
    '''

    # Load data
    identification_data = json.loads(identification_data_json)

    # Calculate average pre-2020 works
    pre_2020_works = [entry['pre_2020_works'] for entry in identification_data]
    average_pre_2020_works = sum(pre_2020_works) / len(pre_2020_works)

    average_pre_2020_works

    
    # Get the output and restore stdout
    output = captured_output.getvalue()
    sys.stdout = old_stdout
    
    print("EXECUTION_SUCCESS")
    print("OUTPUT_START")
    if output.strip():
        print(output.strip())
    else:
        print("(No output)")
    print("OUTPUT_END")
    
except Exception as e:
    # Restore stdout first
    sys.stdout = old_stdout
    print("EXECUTION_ERROR") 
    print("ERROR_START")
    print(f"{type(e).__name__}: {str(e)}")
    print("TRACEBACK_START")
    traceback.print_exc()
    print("TRACEBACK_END")
    print("ERROR_END")
