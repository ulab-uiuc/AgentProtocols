
import sys
import traceback
import os
from io import StringIO
import pandas

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
import PyPDF2

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text() + '\n'
            return text
    except Exception as e:
        return str(e)

# Extract text from the given PDF file
pdf_text = extract_text_from_pdf('67e8878b-5cef-4375-804e-e6291fdbe78a.pdf')
print(pdf_text)
'''.strip()

    # 尝试先将代码作为表达式进行 eval，以便捕获表达式的返回值（例如 DataFrame.head())
    try:
        try:
            compiled_expr = compile(full_code, '<string>', 'eval')
        except SyntaxError:
            compiled_expr = None

        if compiled_expr is not None:
            # 是单个表达式，使用 eval 获取返回值并尝试以友好形式打印
            result_value = eval(compiled_expr, exec_globals)
            try:
                # pandas DataFrame / Series 的友好打印
                if hasattr(result_value, 'to_string'):
                    print(result_value.to_string())
                # numpy arrays 或其他有 shape 的对象，尽量用 repr
                elif hasattr(result_value, '__array__') or hasattr(result_value, 'shape'):
                    try:
                        import numpy as _np
                        # 尝试限制输出长度
                        print(repr(result_value))
                    except Exception:
                        print(repr(result_value))
                else:
                    print(repr(result_value))
            except Exception:
                # 即使格式化失败，也确保有输出
                print(repr(result_value))
        else:
            # 不是表达式，作为普通脚本执行（保留原有行为）
            exec(compile(full_code, '<string>', 'exec'), exec_globals)

    except Exception:
        # 抛出异常以便外层捕获并触发智能修复流程
        raise

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
