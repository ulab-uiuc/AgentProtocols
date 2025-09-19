#!/usr/bin/env python3
"""
Biopython Test for Sandbox Python Execute Tool

This test recreates the exact scenario from the failed GAIA task to verify 
that the sandbox can handle Biopython dependencies and PDB file processing.
"""
import asyncio
import sys
import os
from pathlib import Path

# Add the gaia directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

async def test_biopython_case():
    """Test the exact Biopython case that failed in the original GAIA task."""
    print("=" * 80)
    print("🧬 Testing Biopython PDB Processing in Sandbox")
    print("=" * 80)
    
    try:
        # Import the sandbox tool
        from tools.sandbox_python_execute import SandboxPythonExecute
        
        print("✅ Successfully imported SandboxPythonExecute")
        
        # Create the tool instance
        tool = SandboxPythonExecute()
        
        # The exact code that failed in the original GAIA task
        biopython_code = """
# Original GAIA task: Parse PDB file 5wb7 and calculate distance between first two atoms
from Bio.PDB import PDBParser
import numpy as np
import urllib.request
import os

print("🔬 Starting Biopython PDB analysis...")

# Download the PDB file for protein 5wb7
pdb_id = "5wb7"
pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
pdb_filename = f"{pdb_id}.pdb"

print(f"📥 Downloading PDB file from: {pdb_url}")
try:
    urllib.request.urlretrieve(pdb_url, pdb_filename)
    print(f"✅ Successfully downloaded {pdb_filename}")
    
    # Verify file exists and check size
    if os.path.exists(pdb_filename):
        file_size = os.path.getsize(pdb_filename)
        print(f"📁 File size: {file_size} bytes")
    else:
        print("❌ File download failed")
        exit(1)
        
except Exception as e:
    print(f"❌ Download failed: {e}")
    exit(1)

# Parse the PDB file
print("🔍 Parsing PDB structure...")
parser = PDBParser(QUIET=True)
try:
    structure = parser.get_structure(pdb_id, pdb_filename)
    print("✅ Successfully parsed PDB structure")
except Exception as e:
    print(f"❌ PDB parsing failed: {e}")
    exit(1)

# Get all atoms from the structure
print("⚛️  Extracting atoms from structure...")
all_atoms = list(structure.get_atoms())
total_atoms = len(all_atoms)
print(f"📊 Total atoms in structure: {total_atoms}")

if total_atoms < 2:
    print("❌ Error: Structure has fewer than 2 atoms")
    exit(1)

# Get first and second atoms
atom1 = all_atoms[0]
atom2 = all_atoms[1]

print(f"🎯 First atom:")
print(f"   - Name: {atom1.get_name()}")
print(f"   - Residue: {atom1.get_parent().get_resname()}")
print(f"   - Chain: {atom1.get_parent().get_parent().get_id()}")

print(f"🎯 Second atom:")
print(f"   - Name: {atom2.get_name()}")
print(f"   - Residue: {atom2.get_parent().get_resname()}")
print(f"   - Chain: {atom2.get_parent().get_parent().get_id()}")

# Calculate distance between the atoms
coord1 = atom1.get_coord()
coord2 = atom2.get_coord()

print(f"📍 Coordinates of first atom: [{coord1[0]:.3f}, {coord1[1]:.3f}, {coord1[2]:.3f}]")
print(f"📍 Coordinates of second atom: [{coord2[0]:.3f}, {coord2[1]:.3f}, {coord2[2]:.3f}]")

# Calculate Euclidean distance
distance = np.linalg.norm(coord1 - coord2)
print(f"📏 Raw distance: {distance} Angstroms")

# Round to nearest picometer (3 decimal places in Angstroms)
rounded_distance = round(distance, 3)

print("🎊 CALCULATION COMPLETE!")
print("=" * 50)
print(f"🏆 FINAL ANSWER: {rounded_distance} Angstroms")
print("=" * 50)

# Additional verification
print("\\n🔍 Additional verification:")
print(f"Distance calculation method: Euclidean norm")
print(f"Precision: rounded to nearest picometer (0.001 Å)")
print(f"PDB file source: RCSB Protein Data Bank")
print(f"Protein ID: {pdb_id}")

print("\\n✅ Analysis completed successfully!")
"""
        
        print("🚀 Starting Biopython analysis in sandbox...")
        print("📦 Installing required packages: biopython, numpy")
        
        # Execute with required packages
        result = await tool.execute(
            code=biopython_code,
            packages=["biopython", "numpy"],
            timeout=180  # Allow enough time for package installation and download
        )
        
        print("\n" + "=" * 80)
        print("📊 EXECUTION RESULTS")
        print("=" * 80)
        
        if result.error:
            print("❌ EXECUTION FAILED:")
            print(result.error)
            return False
        else:
            print("✅ EXECUTION SUCCESSFUL:")
            print("\n" + result.output)
            
            # Check if we got the final answer
            if "FINAL ANSWER:" in result.output:
                print("\n🎉 SUCCESS: Found the final answer in output!")
                
                # Extract the answer
                lines = result.output.split('\n')
                for line in lines:
                    if "FINAL ANSWER:" in line:
                        print(f"🏆 {line.strip()}")
                        break
                return True
            else:
                print("\n⚠️  WARNING: No final answer found in output")
                return False
                
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("Make sure the sandbox tools are properly installed")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        try:
            if 'tool' in locals():
                await tool.cleanup()
                print("🧹 Sandbox cleaned up successfully")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")

async def test_basic_sandbox_functionality():
    """Test basic sandbox functionality before the complex Biopython case."""
    print("\n" + "=" * 80)
    print("🔧 Testing Basic Sandbox Functionality")
    print("=" * 80)
    
    try:
        from tools.sandbox_python_execute import SandboxPythonExecute
        
        tool = SandboxPythonExecute()
        
        # Test 1: Basic Python
        print("\n📋 Test 1: Basic Python execution")
        result1 = await tool.execute("""
import sys
import os
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print("Basic Python functionality: ✅")
""")
        
        if result1.error:
            print(f"❌ Basic test failed: {result1.error}")
            return False
        else:
            print("✅ Basic test passed")
            
        # Test 2: Package installation
        print("\n📋 Test 2: Package installation (requests)")
        result2 = await tool.execute("""
import requests
print(f"Requests version: {requests.__version__}")
print("Package installation: ✅")

# Test a simple HTTP request
try:
    response = requests.get("https://httpbin.org/json", timeout=10)
    print(f"HTTP test status: {response.status_code}")
    print("Network access: ✅")
except Exception as e:
    print(f"Network test failed: {e}")
""", packages=["requests"])
        
        if result2.error:
            print(f"❌ Package test failed: {result2.error}")
            return False
        else:
            print("✅ Package test passed")
            
        await tool.cleanup()
        print("✅ Basic functionality tests completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

async def main():
    """Main test runner."""
    print("🧪 SANDBOX PYTHON EXECUTE - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    # Test basic functionality first
    basic_success = await test_basic_sandbox_functionality()
    
    if not basic_success:
        print("\n❌ Basic functionality failed. Skipping Biopython test.")
        return
    
    # Test the Biopython case
    print("\n🧬 Proceeding to Biopython test...")
    biopython_success = await test_biopython_case()
    
    # Final summary
    print("\n" + "=" * 80)
    print("📋 FINAL TEST SUMMARY")
    print("=" * 80)
    print(f"Basic functionality: {'✅ PASSED' if basic_success else '❌ FAILED'}")
    print(f"Biopython analysis: {'✅ PASSED' if biopython_success else '❌ FAILED'}")
    
    if basic_success and biopython_success:
        print("\n🎉 ALL TESTS PASSED! The sandbox can handle the original failing GAIA task.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
    
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())