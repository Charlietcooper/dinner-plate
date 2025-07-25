#!/usr/bin/env python3
"""
Setup script for Constellation Annotator

This script helps with installation and configuration of the constellation annotator.
"""

import os
import sys
import subprocess
import platform

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} - Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} - Failed")
        print(f"  Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 7:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.7+")
        return False

def install_dependencies():
    """Install required dependencies."""
    print("\nInstalling dependencies...")
    
    # Upgrade pip first
    run_command("python -m pip install --upgrade pip", "Upgrading pip")
    
    # Install requirements
    success = run_command("pip install -r requirements.txt", "Installing requirements")
    
    if success:
        print("✓ All dependencies installed successfully")
    else:
        print("✗ Some dependencies failed to install")
        print("You may need to install them manually:")
        print("  pip install astropy astroquery opencv-python Pillow numpy requests")
    
    return success

def test_installation():
    """Test if the installation works."""
    print("\nTesting installation...")
    
    try:
        # Test imports
        import numpy
        import cv2
        from PIL import Image
        import requests
        from astropy.io import fits
        from astropy.wcs import WCS
        from astroquery.astrometry_net import AstrometryNet
        
        print("✓ All core libraries imported successfully")
        
        # Test constellation annotator
        from constellation_annotator import ConstellationAnnotator
        print("✓ Constellation annotator imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def setup_api_key():
    """Help user set up API key."""
    print("\nSetting up Astrometry.net API key...")
    
    # Check if already set
    api_key = os.getenv('ASTROMETRY_API_KEY')
    if api_key:
        print(f"✓ API key already set: {api_key[:8]}...")
        return True
    
    print("You need an Astrometry.net API key for plate solving.")
    print("1. Visit https://nova.astrometry.net/")
    print("2. Create a free account")
    print("3. Go to your profile and copy your API key")
    print("4. Set it as an environment variable:")
    print("   export ASTROMETRY_API_KEY='your_api_key_here'")
    
    # Try to help with platform-specific setup
    system = platform.system().lower()
    
    if system == "linux" or system == "darwin":
        print("\nFor Linux/macOS, add to your shell profile (.bashrc, .zshrc, etc.):")
        print("echo 'export ASTROMETRY_API_KEY=\"your_api_key_here\"' >> ~/.bashrc")
        print("source ~/.bashrc")
    elif system == "windows":
        print("\nFor Windows, set environment variable:")
        print("setx ASTROMETRY_API_KEY \"your_api_key_here\"")
    
    return False

def create_sample_script():
    """Create a sample script for testing."""
    print("\nCreating sample test script...")
    
    sample_script = '''#!/usr/bin/env python3
"""
Sample test script for constellation annotator
"""

import os
from constellation_annotator import ConstellationAnnotator

def test_basic_functionality():
    """Test basic functionality without processing an image."""
    
    # Check API key
    api_key = os.getenv('ASTROMETRY_API_KEY')
    if not api_key:
        print("⚠ No API key found. Set ASTROMETRY_API_KEY environment variable.")
        return False
    
    # Create annotator
    annotator = ConstellationAnnotator(api_key)
    
    # Test constellation data
    constellations = annotator.constellation_data.get_all_constellations()
    print(f"✓ Loaded {len(constellations)} constellations")
    
    # Test star data
    stars = list(annotator.constellation_data.bright_stars.keys())
    print(f"✓ Loaded {len(stars)} bright stars")
    
    print("✓ Basic functionality test passed!")
    return True

if __name__ == "__main__":
    test_basic_functionality()
'''
    
    try:
        with open("test_basic.py", "w") as f:
            f.write(sample_script)
        print("✓ Created test_basic.py")
        return True
    except Exception as e:
        print(f"✗ Failed to create test script: {e}")
        return False

def main():
    """Main setup function."""
    print("Constellation Annotator Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        print("\nPlease upgrade to Python 3.7 or higher.")
        return False
    
    # Install dependencies
    if not install_dependencies():
        print("\nFailed to install dependencies. Please install manually.")
        return False
    
    # Test installation
    if not test_installation():
        print("\nInstallation test failed. Please check the error messages above.")
        return False
    
    # Setup API key
    api_key_ok = setup_api_key()
    
    # Create sample script
    create_sample_script()
    
    # Final instructions
    print("\n" + "=" * 40)
    print("SETUP COMPLETE!")
    print("\nNext steps:")
    print("1. Get an API key from https://nova.astrometry.net/")
    print("2. Set it as environment variable")
    print("3. Test with: python test_basic.py")
    print("4. Process an image: python constellation_annotator.py input.jpg output.jpg")
    
    if api_key_ok:
        print("\n✓ Ready to use!")
    else:
        print("\n⚠ Remember to set your API key before processing images.")
    
    return True

if __name__ == "__main__":
    main() 