#!/usr/bin/env python3
"""
Simple Docker connectivity test for sandbox functionality
"""
import sys
from pathlib import Path

# Add the gaia directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_docker_connectivity():
    """Test basic Docker connectivity and functionality."""
    print("=" * 60)
    print("🐳 Testing Docker Connectivity")
    print("=" * 60)
    
    try:
        import docker
        print("✅ Docker library imported successfully")
        
        # Test Docker daemon connectivity
        client = docker.from_env()
        print("✅ Docker client created successfully")
        
        # Test Docker daemon ping
        client.ping()
        print("✅ Docker daemon is responding")
        
        # List existing images
        images = client.images.list()
        print(f"📦 Found {len(images)} images in local Docker registry")
        
        for image in images[:5]:  # Show first 5 images
            tags = image.tags if image.tags else ['<none>:<none>']
            print(f"   - {tags[0]}")
        
        # Test pulling a small image
        print("\n📥 Testing image pull (pulling hello-world image)...")
        try:
            image = client.images.pull("hello-world:latest")
            print("✅ Successfully pulled hello-world image")
            
            # Test running a container
            print("🚀 Testing container creation and execution...")
            container = client.containers.run(
                "hello-world:latest",
                remove=True,
                detach=False
            )
            
            print("✅ Container executed successfully")
            return True
                
        except Exception as e:
            print(f"❌ Container test failed: {e}")
            return False
            
    except docker.errors.DockerException as e:
        print(f"❌ Docker error: {e}")
        return False
    except ImportError:
        print("❌ Docker library not installed")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_python_image_availability():
    """Test if Python images can be pulled and used."""
    print("\n" + "=" * 60)
    print("🐍 Testing Python Image Availability")
    print("=" * 60)
    
    try:
        import docker
        client = docker.from_env()
        
        # List of Python images to try (from most to least preferred)
        python_images = [
            "python:3.11-slim",
            "python:3.10-slim", 
            "python:3.9-slim",
            "python:3.11",
            "python:3.10",
            "python:latest"
        ]
        
        for image_name in python_images:
            print(f"\n📋 Testing {image_name}...")
            try:
                # Try to pull the image
                print(f"📥 Pulling {image_name}...")
                image = client.images.pull(image_name)
                print(f"✅ Successfully pulled {image_name}")
                
                # Test running Python in the container
                print(f"🐍 Testing Python execution in {image_name}...")
                container = client.containers.run(
                    image_name,
                    ["python", "-c", "import sys; print(f'Python {sys.version_info.major}.{sys.version_info.minor}')"],
                    remove=True,
                    detach=False,
                    stdout=True,
                    stderr=True
                )
                
                print(f"✅ Python container executed successfully")
                print(f"🎉 {image_name} is working and ready to use!")
                return image_name
                
            except docker.errors.ImageNotFound:
                print(f"❌ Image {image_name} not found")
                continue
            except Exception as e:
                print(f"❌ Error with {image_name}: {e}")
                continue
        
        print("\n❌ No working Python images found")
        return None
        
    except Exception as e:
        print(f"❌ Python image test failed: {e}")
        return None

if __name__ == "__main__":
    print("🧪 DOCKER SANDBOX CONNECTIVITY TEST")
    print("=" * 60)
    
    # Test basic Docker connectivity
    docker_ok = test_docker_connectivity()
    
    if not docker_ok:
        print("\n❌ Docker connectivity failed. Cannot proceed with sandbox tests.")
        sys.exit(1)
    
    # Test Python image availability
    working_image = test_python_image_availability()
    
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    print(f"Docker connectivity: {'✅ OK' if docker_ok else '❌ FAILED'}")
    print(f"Python image available: {'✅ ' + working_image if working_image else '❌ NONE'}")
    
    if docker_ok and working_image:
        print(f"\n🎉 SUCCESS! Ready to use Docker sandbox with {working_image}")
        print(f"💡 Update your sandbox configuration to use: {working_image}")
    else:
        print("\n⚠️  Some issues detected. Check Docker installation and network connectivity.")
    
    print("=" * 60)