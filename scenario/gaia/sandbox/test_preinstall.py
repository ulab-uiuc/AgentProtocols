
import asyncio
import sys
from pathlib import Path

# This script needs to be run from the project root or have its path handled correctly.
# To ensure modules are found, we add the 'script/gaia' directory to the Python path.
# The script is in .../script/gaia/sandbox/, so two parents up is .../script/gaia.
GAIA_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GAIA_ROOT))

try:
    from sandbox.core.sandbox import DockerSandbox
    from tools.utils.config import SandboxSettings
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please make sure you are running this script from a location where the necessary modules can be found.")
    print("This script assumes it is located in 'script/gaia/sandbox/' and adds the 'script/gaia' directory to the path.")
    sys.exit(1)

async def main():
    """
    Initializes a Docker sandbox, checks if pandas is pre-installed, and cleans up.
    """
    print("--- Starting Sandbox Pre-install Test ---")
    
    # 1. Configure sandbox settings.
    # We must enable the network to allow the initial `pip install pandas` to run,
    # which happens automatically during sandbox creation if network is on.
    settings = SandboxSettings(
        use_sandbox=True,
        network_enabled=True,
        image="python:3.11-slim"  # Match the image we configured
    )
    
    sandbox = DockerSandbox(config=settings)
    
    try:
        # 2. Create and initialize the sandbox.
        # The `create()` method will automatically attempt to install pandas.
        print(f"Creating sandbox with image '{settings.image}' and network enabled...")
        await sandbox.create()
        print("Sandbox created successfully.")
        
        # 3. Run the test command to verify pandas installation.
        print("\nAttempting to import pandas inside the sandbox...")
        command_to_run = 'python -c "import pandas; print(f\'Successfully imported pandas version: {pandas.__version__}\')"'
        
        try:
            result = await sandbox.run_command(command_to_run, timeout=60)
            
            print("\n--- TEST RESULT ---")
            print(result.strip())
            print("-------------------\n")
            
            if "Successfully imported pandas" in result:
                print("✅ Test PASSED: pandas is correctly installed and importable in the sandbox.")
            else:
                print("❌ Test FAILED: Could not verify pandas installation. See output above for details.")
                
        except Exception as e:
            print(f"\n❌ An error occurred while running the test command: {e}")
            print("❌ Test FAILED.")

    except Exception as e:
        print(f"\n❌ An error occurred during sandbox setup: {e}")
        print("❌ Test FAILED.")
        
    finally:
        # 4. Clean up the sandbox container.
        print("\nCleaning up sandbox container...")
        if sandbox.container:
            await sandbox.cleanup()
            print("Cleanup complete.")
        else:
            print("No container was created, skipping cleanup.")
            
    print("\n--- Test Finished ---")

if __name__ == "__main__":
    # This allows running the script directly with `python test_preinstall.py`
    asyncio.run(main())
