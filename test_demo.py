#!/usr/bin/env python3
"""
Demo script for Dinner Plate constellation annotator
Shows the constellation data and functionality without requiring an API key.
"""

from constellation_annotator import ConstellationAnnotator
from constellation_data_enhanced import EnhancedConstellationData
import numpy as np

def demo_constellation_data():
    """Demonstrate the constellation data."""
    print("🌟 Dinner Plate Constellation Annotator Demo")
    print("=" * 50)
    
    # Load constellation data
    data = EnhancedConstellationData()
    
    print(f"📊 Loaded {len(data.get_all_constellations())} constellations")
    print(f"⭐ Loaded {len(data.bright_stars)} bright stars")
    
    # Show some example constellations
    print("\n🔭 Example Constellations:")
    example_constellations = ['Orion', 'Ursa Major', 'Cassiopeia', 'Scorpius', 'Cygnus']
    
    for constellation in example_constellations:
        stars = data.get_constellation_stars(constellation)
        print(f"\n{constellation}:")
        for i, (star1, star2) in enumerate(stars[:3]):  # Show first 3 lines
            coords1 = data.get_star_coordinates(star1)
            coords2 = data.get_star_coordinates(star2)
            if coords1 and coords2:
                print(f"  Line {i+1}: {star1} → {star2}")
                print(f"    {star1}: RA {coords1[0]:.2f}°, Dec {coords1[1]:.2f}°")
                print(f"    {star2}: RA {coords2[0]:.2f}°, Dec {coords2[1]:.2f}°")
    
    # Show constellations by hemisphere
    northern = data.get_constellations_by_hemisphere('north')
    southern = data.get_constellations_by_hemisphere('south')
    
    print(f"\n🌍 Constellation Distribution:")
    print(f"  Northern Hemisphere: {len(northern)} constellations")
    print(f"  Southern Hemisphere: {len(southern)} constellations")
    
    return data

def demo_annotator_functionality():
    """Demonstrate the annotator functionality."""
    print("\n🎨 Annotator Functionality Demo:")
    print("-" * 30)
    
    # Create annotator (without API key for demo)
    annotator = ConstellationAnnotator()
    
    print(f"📐 Drawing parameters:")
    print(f"  Line color: {annotator.line_color} (RGB)")
    print(f"  Line thickness: {annotator.line_thickness} pixels")
    print(f"  Star color: {annotator.star_color} (RGB)")
    print(f"  Star radius: {annotator.star_radius} pixels")
    print(f"  Text color: {annotator.text_color} (RGB)")
    print(f"  Font size: {annotator.font_size} points")
    
    # Show how to customize
    print(f"\n🎛️  Customization example:")
    print(f"  # Change line color to red")
    print(f"  annotator.line_color = (255, 0, 0)")
    print(f"  # Change line thickness")
    print(f"  annotator.line_thickness = 3")
    
    return annotator

def demo_coordinate_conversion():
    """Demonstrate coordinate conversion (simulated)."""
    print("\n🌌 Coordinate Conversion Demo:")
    print("-" * 30)
    
    # Simulate some star coordinates
    test_stars = [
        ("Betelgeuse", (88.7929, 7.4071)),
        ("Rigel", (78.6345, -8.2016)),
        ("Sirius", (101.2872, -16.7161)),
        ("Polaris", (37.9529, 89.2642))
    ]
    
    print("Example star coordinates (RA, Dec):")
    for star_name, (ra, dec) in test_stars:
        print(f"  {star_name}: RA {ra:.2f}°, Dec {dec:.2f}°")
    
    print(f"\n📐 The script converts these sky coordinates to pixel coordinates")
    print(f"   using plate solving from Astrometry.net")
    
    return test_stars

def demo_usage_instructions():
    """Show usage instructions."""
    print("\n📖 Usage Instructions:")
    print("-" * 20)
    
    print("1. Get API Key:")
    print("   Visit: https://nova.astrometry.net/")
    print("   Create free account and get API key")
    
    print("\n2. Set Environment Variable:")
    print("   export ASTROMETRY_API_KEY='your_api_key_here'")
    
    print("\n3. Process an Image:")
    print("   python constellation_annotator.py input.jpg output.jpg")
    
    print("\n4. With API Key:")
    print("   python constellation_annotator.py input.jpg output.jpg --api-key YOUR_KEY")
    
    print("\n5. Verbose Output:")
    print("   python constellation_annotator.py input.jpg output.jpg --verbose")

def main():
    """Run the complete demo."""
    try:
        # Run all demos
        data = demo_constellation_data()
        annotator = demo_annotator_functionality()
        test_stars = demo_coordinate_conversion()
        demo_usage_instructions()
        
        print("\n" + "=" * 50)
        print("✅ Demo completed successfully!")
        print("\n🎯 Ready to process your astronomical images!")
        print("   Just get an API key and try it with your Milky Way photo!")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        print("Please check the installation and try again.")

if __name__ == "__main__":
    main() 