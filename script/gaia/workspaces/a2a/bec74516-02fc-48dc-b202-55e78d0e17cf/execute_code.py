
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
    import requests
    import json
    import sys

    # List of ORCID IDs from JSON-LD file
    dataset = {
        "contributors": [
            "0000-0003-0396-0333",
            "0000-0002-2605-6569",
            "0000-0001-6102-7846",
            "0000-0002-0209-2784",
            "0000-0002-1053-2030"
        ]
    }

    # ORCID API base URL
    base_url = "https://pub.orcid.org/v3.0/"

    # Headers for ORCID API requests
    def get_headers():
        return {
            'Accept': 'application/json'
        }

    # Function to fetch works for a given ORCID ID
    def fetch_works(orcid_id):
        url = f"{base_url}{orcid_id}/works"
        response = requests.get(url, headers=get_headers())
        response.raise_for_status()
        return response.json()

    # Function to count pre-2020 works
    def count_pre_2020_works(works):
        count = 0
        for group in works["group"]:
            for work_summary in group["work-summary"]:
                if "publication-date" in work_summary and "year" in work_summary["publication-date"]:
                    year = int(work_summary["publication-date"]["year"]["value"])
                    if year < 2020:
                        count += 1
        return count

    # Fetch and analyze works for each contributor total pre-2020 works
    pre_2020_works_counts = []

    for orcid_id in dataset["contributors"]:
        try:
            works = fetch_works(orcid_id)
            pre_2020_count = count_pre_2020_works(works)
            pre_2020_works_counts.append(pre_2020_count)
        except Exception as e:
            print(f"Error fetching data for {orcid_id}: {e}", file=sys.stderr)

    # Calculate the average number of works
    if pre_2020_works_counts:
        average_pre_2020_works = sum(pre_2020_works_counts) / len(pre_2020_works_counts)
    else:
        average_pre_2020_works = 0

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
