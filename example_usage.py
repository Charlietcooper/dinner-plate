#!/usr/bin/env python3
"""
Example usage of the Constellation Annotator

This script demonstrates how to use the constellation annotator with different
options and provides examples of error handling.
"""

import os
import sys
from pathlib import Path

# Import the constellation annotator
try:
    from constellation_annotator import ConstellationAnnotator
    from constellation_data_enhanced import EnhancedConstellationData
except ImportError as e:
    print(f"Error importing constellation annotator: {e}")
    print("Make sure you have installed all dependencies:")
    print("pip install -r requirements.txt")
    sys.exit(1)

def example_basic_usage():
    """Example of basic usage with error handling."""
    print("=== Basic Usage Example ===")
    
    # Check if API key is available
    api_key = os.getenv('ASTROMETRY_API_KEY')
    if not api_key:
        print("⚠ No API key found. Set ASTROMETRY_API_KEY environment variable.")
        print("Get a free key at: https://nova.astrometry.net/")
        return False
    
    # Create annotator
    annotator = ConstellationAnnotator(api_key)
    
    # Example input and output paths
    input_image = "example_image.jpg"  # Replace with your image
    output_image = "example_annotated.jpg"
    
    # Check if input file exists
    if not os.path.exists(input_image):
        print(f"⚠ Input image not found: {input_image}")
        print("Please provide a valid astronomical image file.")
        return False
    
    print(f"Processing {input_image}...")
    
    # Process the image
    success = annotator.annotate_image(input_image, output_image)
    
    if success:
        print(f"✓ Successfully created annotated image: {output_image}")
        return True
    else:
        print("✗ Failed to annotate image")
        return False

def example_custom_parameters():
    """Example with custom drawing parameters."""
    print("\n=== Custom Parameters Example ===")
    
    api_key = os.getenv('ASTROMETRY_API_KEY')
    if not api_key:
        print("⚠ No API key available for this example")
        return False
    
    # Create annotator
    annotator = ConstellationAnnotator(api_key)
    
    # Customize drawing parameters
    annotator.line_color = (255, 0, 0)      # Red lines
    annotator.line_thickness = 3            # Thicker lines
    annotator.star_color = (0, 255, 255)    # Cyan stars
    annotator.star_radius = 5               # Larger stars
    annotator.text_color = (255, 255, 0)    # Yellow text
    annotator.font_size = 20                # Larger font
    
    print("Customized drawing parameters:")
    print(f"  Line color: {annotator.line_color}")
    print(f"  Line thickness: {annotator.line_thickness}")
    print(f"  Star color: {annotator.star_color}")
    print(f"  Star radius: {annotator.star_radius}")
    print(f"  Text color: {annotator.text_color}")
    print(f"  Font size: {annotator.font_size}")
    
    return True

def example_constellation_data():
    """Example of working with constellation data."""
    print("\n=== Constellation Data Example ===")
    
    # Create enhanced constellation data
    data = EnhancedConstellationData()
    
    print(f"Available constellations: {len(data.get_all_constellations())}")
    
    # Show some example constellations
    example_constellations = ['Orion', 'Ursa Major', 'Cassiopeia', 'Scorpius']
    
    for constellation in example_constellations:
        stars = data.get_constellation_stars(constellation)
        print(f"\n{constellation}:")
        for star1, star2 in stars[:3]:  # Show first 3 lines
            coords1 = data.get_star_coordinates(star1)
            coords2 = data.get_star_coordinates(star2)
            if coords1 and coords2:
                print(f"  {star1} ({coords1[0]:.2f}°, {coords1[1]:.2f}°) -> "
                      f"{star2} ({coords2[0]:.2f}°, {coords2[1]:.2f}°)")
    
    # Show constellations by hemisphere
    northern = data.get_constellations_by_hemisphere('north')
    southern = data.get_constellations_by_hemisphere('south')
    
    print(f"\nNorthern hemisphere constellations: {len(northern)}")
    print(f"Southern hemisphere constellations: {len(southern)}")
    
    return True

def example_batch_processing():
    """Example of batch processing multiple images."""
    print("\n=== Batch Processing Example ===")
    
    api_key = os.getenv('ASTROMETRY_API_KEY')
    if not api_key:
        print("⚠ No API key available for batch processing")
        return False
    
    # Example image files (replace with your actual files)
    image_files = [
        "image1.jpg",
        "image2.jpg", 
        "image3.jpg"
    ]
    
    # Filter to only existing files
    existing_files = [f for f in image_files if os.path.exists(f)]
    
    if not existing_files:
        print("⚠ No image files found for batch processing")
        print("Please provide valid image files.")
        return False
    
    print(f"Found {len(existing_files)} images to process")
    
    # Create annotator
    annotator = ConstellationAnnotator(api_key)
    
    # Process each image
    success_count = 0
    for input_file in existing_files:
        output_file = f"annotated_{Path(input_file).name}"
        
        print(f"\nProcessing {input_file}...")
        success = annotator.annotate_image(input_file, output_file)
        
        if success:
            print(f"✓ Created {output_file}")
            success_count += 1
        else:
            print(f"✗ Failed to process {input_file}")
    
    print(f"\nBatch processing complete: {success_count}/{len(existing_files)} successful")
    return success_count > 0

def example_error_handling():
    """Example of error handling and troubleshooting."""
    print("\n=== Error Handling Example ===")
    
    # Test without API key
    print("Testing without API key...")
    annotator_no_key = ConstellationAnnotator()
    
    # This would fail, but we can still test the constellation data
    data = annotator_no_key.constellation_data
    print(f"✓ Constellation data loaded: {len(data.get_all_constellations())} constellations")
    
    # Test with invalid file
    print("\nTesting with invalid file...")
    success = annotator_no_key.annotate_image("nonexistent_file.jpg", "output.jpg")
    if not success:
        print("✓ Correctly handled invalid file")
    
    # Test image loading
    print("\nTesting image loading...")
    image = annotator_no_key.load_image("nonexistent_file.jpg")
    if image is None:
        print("✓ Correctly handled image loading error")
    
    return True

def main():
    """Run all examples."""
    print("Constellation Annotator - Example Usage")
    print("=" * 50)
    
    # Run examples
    examples = [
        ("Basic Usage", example_basic_usage),
        ("Custom Parameters", example_custom_parameters),
        ("Constellation Data", example_constellation_data),
        ("Batch Processing", example_batch_processing),
        ("Error Handling", example_error_handling),
    ]
    
    results = []
    for name, example_func in examples:
        try:
            print(f"\n{'='*20} {name} {'='*20}")
            result = example_func()
            results.append((name, result))
        except Exception as e:
            print(f"Error in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("EXAMPLE SUMMARY:")
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {name}: {status}")
    
    print(f"\nTo use the constellation annotator:")
    print("1. Get an API key from https://nova.astrometry.net/")
    print("2. Set it as environment variable: export ASTROMETRY_API_KEY='your_key'")
    print("3. Run: python constellation_annotator.py input.jpg output.jpg")

if __name__ == "__main__":
    main() 