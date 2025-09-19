
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

    def calculate_distance(atom1, atom2):
        coord1 = np.array(atom1.coord)
        coord2 = np.array(atom2.coord)
        return np.linalg.norm(coord1 - coord2)

    # Initialize PDB parser
    parser = PDBParser(QUIET=True)

    # Parse the PDB file
    structure = parser.get_structure('5wb7', '7dd30055-0198-452e-8c25-f73dbe27dcb8.pdb')

    # Extract atoms
    atoms = list(structure.get_atoms())

    if len(atoms) >= 2:
        # Calculate the distance between the first and second atom
        dist = calculate_distance(atoms[0], atoms[1])
        # Round the distance to the nearest picometer (1 Angstrom = 10000 picometers)
        dist_picometers = round(dist * 10000)
        output = f"Distance between first and second atom: {dist_picometers} pm"
    else:
        output = "Not enough atoms in the structure to calculate the distance."

    output
    
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
