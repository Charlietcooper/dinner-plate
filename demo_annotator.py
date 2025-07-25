#!/usr/bin/env python3
"""
Demo script showing constellation annotation with simulated plate solving
"""

import cv2
import numpy as np
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from constellation_data_enhanced import EnhancedConstellationData
import os

def load_original_image(image_path="Input/test-1.jpg"):
    """Load the original image as background."""
    if os.path.exists(image_path):
        print(f"📸 Loading original image: {image_path}")
        image = cv2.imread(image_path)
        if image is not None:
            print(f"✅ Loaded image: {image.shape[1]} x {image.shape[0]} pixels")
            return image
        else:
            print(f"❌ Failed to load image: {image_path}")
    else:
        print(f"❌ Image not found: {image_path}")
    
    # Fallback to creating demo image if original not available
    print("🔄 Falling back to demo image...")
    return create_demo_image(2048, 1365)

def create_demo_image(width=2048, height=1365):
    """Create a demo image with some bright stars."""
    print(f"🌟 Creating demo star field: {width}x{height}")
    
    # Create black background
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add some background noise
    noise = np.random.normal(5, 2, (height, width, 3)).astype(np.uint8)
    image = np.clip(image + noise, 0, 255)
    
    # Add some background stars
    for _ in range(500):
        x = np.random.randint(50, width-50)
        y = np.random.randint(50, height-50)
        brightness = np.random.randint(30, 150)
        cv2.circle(image, (x, y), 2, (brightness, brightness, brightness), -1)
    
    return image

def create_demo_wcs(width, height):
    """Create a demo WCS for the image."""
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [width//2, height//2]  # Reference pixel (center)
    wcs.wcs.crval = [180, 0]    # Reference coordinates (RA=180°, Dec=0°)
    wcs.wcs.cdelt = [0.1, 0.1]  # Pixel scale (degrees per pixel)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs

def sky_to_pixel(wcs, ra, dec):
    """Convert sky coordinates to pixel coordinates."""
    try:
        coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
        x, y = wcs.world_to_pixel(coord)
        return float(x), float(y)
    except:
        return None

def draw_constellation_lines(image, wcs, constellation_data):
    """Draw constellation lines on the image."""
    lines_drawn = 0
    constellation_positions = {}  # Track constellation positions for labels
    
    for constellation_name, line_pairs in constellation_data.constellation_lines.items():
        constellation_visible = False
        constellation_center_x = 0
        constellation_center_y = 0
        visible_lines = 0
        
        for star1_name, star2_name in line_pairs:
            # Get star coordinates
            star1_coords = constellation_data.bright_stars.get(star1_name)
            star2_coords = constellation_data.bright_stars.get(star2_name)
            
            if star1_coords and star2_coords:
                ra1, dec1 = star1_coords
                ra2, dec2 = star2_coords
                
                # Convert to pixel coordinates
                pixel1 = sky_to_pixel(wcs, ra1, dec1)
                pixel2 = sky_to_pixel(wcs, ra2, dec2)
                
                if pixel1 and pixel2:
                    # Check for NaN values
                    if (np.isnan(pixel1[0]) or np.isnan(pixel1[1]) or 
                        np.isnan(pixel2[0]) or np.isnan(pixel2[1])):
                        continue
                        
                    x1, y1 = int(pixel1[0]), int(pixel1[1])
                    x2, y2 = int(pixel2[0]), int(pixel2[1])
                    
                    # Check if points are within image bounds
                    if (0 <= x1 < image.shape[1] and 0 <= y1 < image.shape[0] and
                        0 <= x2 < image.shape[1] and 0 <= y2 < image.shape[0]):
                        
                        # Draw line
                        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        lines_drawn += 1
                        constellation_visible = True
                        
                        # Track constellation center
                        constellation_center_x += (x1 + x2) / 2
                        constellation_center_y += (y1 + y2) / 2
                        visible_lines += 1
        
        if constellation_visible and visible_lines > 0:
            # Calculate average center position for this constellation
            avg_x = int(constellation_center_x / visible_lines)
            avg_y = int(constellation_center_y / visible_lines)
            constellation_positions[constellation_name] = (avg_x, avg_y)
    
    return lines_drawn, constellation_positions

def draw_bright_stars(image, wcs, constellation_data):
    """Draw bright star markers on the image."""
    stars_drawn = 0
    
    for star_name, coords in constellation_data.bright_stars.items():
        ra, dec = coords
        pixel = sky_to_pixel(wcs, ra, dec)
        
        if pixel:
            # Check for NaN values
            if np.isnan(pixel[0]) or np.isnan(pixel[1]):
                continue
                
            x, y = int(pixel[0]), int(pixel[1])
            
            # Check if point is within image bounds
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                # Draw star marker
                cv2.circle(image, (x, y), 4, (255, 255, 0), -1)
                cv2.circle(image, (x, y), 6, (255, 255, 0), 1)
                stars_drawn += 1
    
    return stars_drawn

def add_constellation_labels(image, constellation_positions):
    """Add constellation labels to the image."""
    try:
        # Convert to PIL for better text handling
        from PIL import Image, ImageDraw, ImageFont
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        # Try to load a font - larger size for better visibility
        try:
            font = ImageFont.truetype("arial.ttf", 32)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 32)
            except:
                font = ImageFont.load_default()
        
        # Track label positions to avoid overlap
        label_positions = []
        
        for constellation_name, (x, y) in constellation_positions.items():
            # Check for overlap with existing labels - adjusted for larger font
            overlap = False
            for pos in label_positions:
                if abs(x - pos[0]) < 120 and abs(y - pos[1]) < 50:
                    overlap = True
                    break
            
            if not overlap:
                # Draw text with offset - adjusted for larger font
                text_x, text_y = x + 15, y - 20
                draw.text((text_x, text_y), constellation_name, 
                        fill=(255, 255, 255), font=font)
                label_positions.append((x, y))
        
        # Convert back to OpenCV format
        image[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
    except Exception as e:
        print(f"Warning: Could not add constellation labels: {e}")

def main():
    """Main demonstration function."""
    print("🌟 Dinner Plate - Constellation Annotator Demo")
    print("=" * 50)
    
    # Load original image as background
    print("📸 Loading original image...")
    image = load_original_image()
    
    # Create demo WCS for the loaded image
    print("🌍 Creating demo coordinate system...")
    height, width = image.shape[:2]
    wcs = create_demo_wcs(width, height)
    
    # Load constellation data
    print("⭐ Loading constellation data...")
    constellation_data = EnhancedConstellationData()
    
    print(f"📊 Loaded {len(constellation_data.constellation_lines)} constellations")
    print(f"⭐ Loaded {len(constellation_data.bright_stars)} bright stars")
    
    # Draw constellation lines
    print("🎨 Drawing constellation lines...")
    lines_drawn, constellation_positions = draw_constellation_lines(image, wcs, constellation_data)
    print(f"📐 Drew {lines_drawn} constellation lines")
    
    # Draw bright star markers
    print("⭐ Drawing bright star markers...")
    stars_drawn = draw_bright_stars(image, wcs, constellation_data)
    print(f"⭐ Drew {stars_drawn} bright star markers")
    
    # Add constellation labels
    print("🏷️ Adding constellation labels...")
    add_constellation_labels(image, constellation_positions)
    print(f"🏷️ Added {len(constellation_positions)} constellation labels")
    
    # Add some text labels
    cv2.putText(image, "Dinner Plate Demo - Original Image", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(image, f"Constellation Lines: {lines_drawn}", (50, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(image, f"Bright Stars: {stars_drawn}", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(image, f"Constellations: {len(constellation_positions)}", (50, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Save the result to Output folder
    output_path = "Output/demo_annotated.jpg"
    success = cv2.imwrite(output_path, image)
    
    if success:
        print(f"\n✅ Demo completed successfully!")
        print(f"📸 Annotated image saved as: {output_path}")
        print(f"📐 Image size: {image.shape[1]} x {image.shape[0]} pixels")
        print(f"📊 Lines drawn: {lines_drawn}")
        print(f"⭐ Stars marked: {stars_drawn}")
        print(f"🏷️ Constellations labeled: {len(constellation_positions)}")
        
        print(f"\n🎯 This demonstrates how the constellation annotator works!")
        print(f"   With real astronomical images, it would:")
        print(f"   1. Use Astrometry.net to determine the image coordinates")
        print(f"   2. Convert constellation star positions to pixel coordinates")
        print(f"   3. Draw the constellation lines and star markers")
        print(f"   4. Add constellation labels")
        
        return True
    else:
        print("❌ Failed to save demo image")
        return False

if __name__ == "__main__":
    main() 