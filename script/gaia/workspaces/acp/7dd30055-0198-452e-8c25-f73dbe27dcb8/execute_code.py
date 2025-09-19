
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
    from Bio.PDB import PDBParser
    import numpy as np

    # Parse the PDB file
    def calculate_distance_between_first_two_atoms(filename):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('5wb7_structure', filename)

        # Extract the first model
        model = next(structure.get_models())

        # Extract the first two atoms
        atom_list = list(model.get_atoms())
        if len(atom_list) < 2:
            raise ValueError("The file does not contain enough atoms to calculate a distance.")

        first_atom = atom_list[0]
        second_atom = atom_list[1]

        # Calculate the distance between the first and second atoms
        distance = np.linalg.norm(first_atom.coord - second_atom.coord)
    
        # Convert distance from angstroms to picometers and round
        distance_in_pm = round(distance * 100, 0)
        return distance_in_pm

    # Path to the PDB file
    pdb_file_path = '7dd30055-0198-452e-8c25-f73dbe27dcb8.pdb'

    # Calculate the distance
    calculate_distance_between_first_two_atoms(pdb_file_path)
    
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
