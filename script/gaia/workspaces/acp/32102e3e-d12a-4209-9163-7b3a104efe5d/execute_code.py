
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
    print(f"Warning: Could not add to __builtins__ ({builtins_error}), using globals fallback")
    globals()['resolve_dataset_file'] = resolve_dataset_file

try:
    # Execute user code
    import pandas as pd

    # Load the Excel file
    try:
        df = pd.read_excel('32102e3e-d12a-4209-9163-7b3a104efe5d.xlsx')
    
        # Filter for Blu-Ray only
        df_blu_ray = df[df['Format'] == 'Blu-Ray']
    
        # Find the row with the oldest release year
        oldest_blu_ray = df_blu_ray.loc[df_blu_ray['Release Year'].idxmin()]
    
        oldest_blu_ray_title = oldest_blu_ray['Title']
        oldest_blu_ray_title
    except Exception as e:
        print(f'Error: {e}')
        print('Attempting to handle the error gracefully...')
    
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
