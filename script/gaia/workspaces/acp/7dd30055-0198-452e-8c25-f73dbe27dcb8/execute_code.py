
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
    from Bio.PDB import PDBParser
    import numpy as np

    # Function to calculate the distance between two atoms given their coordinates
    def calculate_distance(coord1, coord2):
        return np.linalg.norm(coord1 - coord2)

    # Load the PDB file
    pdb_id = "5wb7"
    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure(pdb_id, f'{pdb_id}.pdb')

    # Extract the first two atoms
    atoms = list(structure.get_atoms())

    # Get the coordinates of the first and second atom
    coord1 = atoms[0].get_coord()
    coord2 = atoms[1].get_coord()

    # Calculate the distance
    distance = calculate_distance(coord1, coord2)

    # Round the distance to the nearest picometer (10^-3 Angstrom)
    rounded_distance_in_picometers = round(distance * 1000)

    rounded_distance_in_picometers
    
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
