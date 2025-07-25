#!/usr/bin/env python3
"""
Test script to verify installation of constellation annotator dependencies.
Run this script to check if all required libraries are properly installed.
"""

import sys
import os

def test_import(module_name, package_name=None):
    """Test if a module can be imported."""
    try:
        __import__(module_name)
        print(f"✓ {package_name or module_name} - OK")
        return True
    except ImportError as e:
        print(f"✗ {package_name or module_name} - FAILED: {e}")
        return False

def main():
    """Test all required dependencies."""
    print("Testing Constellation Annotator Dependencies")
    print("=" * 50)
    
    # Core Python libraries
    print("\nCore Python Libraries:")
    test_import("argparse")
    test_import("os")
    test_import("sys")
    test_import("json")
    test_import("logging")
    test_import("pathlib")
    test_import("typing")
    
    # Image processing libraries
    print("\nImage Processing Libraries:")
    numpy_ok = test_import("numpy", "NumPy")
    cv2_ok = test_import("cv2", "OpenCV")
    pil_ok = test_import("PIL", "Pillow")
    requests_ok = test_import("requests", "Requests")
    
    # Astronomy libraries
    print("\nAstronomy Libraries:")
    astropy_ok = test_import("astropy", "Astropy")
    astroquery_ok = test_import("astroquery", "Astroquery")
    
    # Test specific astropy modules
    if astropy_ok:
        print("  Testing Astropy modules:")
        test_import("astropy.io.fits", "Astropy.io.fits")
        test_import("astropy.wcs", "Astropy.wcs")
        test_import("astropy.coordinates", "Astropy.coordinates")
        test_import("astropy.units", "Astropy.units")
    
    if astroquery_ok:
        print("  Testing Astroquery modules:")
        test_import("astroquery.astrometry_net", "Astroquery.astrometry_net")
        test_import("astroquery.vizier", "Astroquery.vizier")
        test_import("astroquery.simbad", "Astroquery.simbad")
    
    # Test constellation annotator
    print("\nConstellation Annotator:")
    try:
        from constellation_annotator import ConstellationAnnotator, ConstellationData, PlateSolver
        print("✓ Constellation Annotator - OK")
        annotator_ok = True
    except ImportError as e:
        print(f"✗ Constellation Annotator - FAILED: {e}")
        annotator_ok = False
    
    # Check API key
    print("\nAPI Key Check:")
    api_key = os.getenv('ASTROMETRY_API_KEY')
    if api_key:
        print(f"✓ Astrometry.net API key found: {api_key[:8]}...")
    else:
        print("⚠ No ASTROMETRY_API_KEY environment variable found")
        print("  You can set it with: export ASTROMETRY_API_KEY='your_key_here'")
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY:")
    
    critical_libs = [numpy_ok, cv2_ok, pil_ok, requests_ok, astropy_ok, astroquery_ok, annotator_ok]
    
    if all(critical_libs):
        print("✓ All critical dependencies are installed!")
        print("\nYou can now use the constellation annotator:")
        print("  python constellation_annotator.py input.jpg output.jpg")
    else:
        print("✗ Some dependencies are missing.")
        print("\nTo install missing dependencies, run:")
        print("  pip install -r requirements.txt")
    
    if not api_key:
        print("\n⚠ Remember to get an API key from https://nova.astrometry.net/")
        print("  and set it as an environment variable.")

if __name__ == "__main__":
    main() 