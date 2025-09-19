
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
    
    # 准备要执行的完整代码
    full_code = '''
import pandas as pd

# Load the spreadsheet to find the oldest Blu-Ray
try:
    file_path = '32102e3e-d12a-4209-9163-7b3a104efe5d.xlsx'
    
    # Read the Excel file
    xls = pd.ExcelFile(file_path)
    
    # Load the first sheet
    sheet_name = xls.sheet_names[0]
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # Filter only Blu-Ray items
    df_blu_ray = df[df['Format'] == 'Blu-Ray']
    
    # Find the oldest Blu-Ray by release year
    oldest_blu_ray = df_blu_ray.loc[df_blu_ray['Release Year'].idxmin()]
    oldest_title = oldest_blu_ray['Title']
    
    oldest_title
except Exception as e:
    print(f'Error: {e}')
    print('Attempting to handle the error gracefully...')
'''.strip()

    # 将整个代码块作为一个整体来执行
    # 这会保留所有的缩进和代码结构
    exec(full_code, exec_globals)
    
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
