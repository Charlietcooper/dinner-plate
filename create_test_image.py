#!/usr/bin/env python3
"""
Create a synthetic astronomical image for testing the constellation annotator
"""

import numpy as np
import cv2
from astropy.coordinates import SkyCoord
from astropy import units as u
import random

def create_synthetic_star_field(width=1024, height=768, num_stars=200):
    """Create a synthetic star field image."""
    
    print(f"🌟 Creating synthetic star field: {width}x{height} with {num_stars} stars")
    
    # Create black background
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add some background noise
    noise = np.random.normal(5, 2, (height, width, 3)).astype(np.uint8)
    image = np.clip(image + noise, 0, 255)
    
    # Generate random star positions
    stars = []
    for i in range(num_stars):
        x = random.randint(50, width - 50)
        y = random.randint(50, height - 50)
        brightness = random.randint(50, 255)
        size = random.randint(1, 3)
        stars.append((x, y, brightness, size))
    
    # Draw stars
    for x, y, brightness, size in stars:
        # Create a small Gaussian star
        for dx in range(-size*2, size*2+1):
            for dy in range(-size*2, size*2+1):
                if 0 <= x + dx < width and 0 <= y + dy < height:
                    distance = np.sqrt(dx*dx + dy*dy)
                    if distance <= size*2:
                        intensity = int(brightness * np.exp(-distance*distance / (2*size*size)))
                        for c in range(3):
                            image[y+dy, x+dx, c] = min(255, image[y+dy, x+dx, c] + intensity)
    
    # Add some constellation-like patterns (simplified)
    # Create a few bright "constellation" stars
    constellation_stars = [
        (width//4, height//4, 255, 4),      # Bright star 1
        (width//4 + 100, height//4 + 50, 255, 3),  # Bright star 2
        (width//4 + 200, height//4 + 100, 255, 3), # Bright star 3
        (width//4 + 150, height//4 + 150, 255, 2), # Bright star 4
        (width//4 + 50, height//4 + 200, 255, 2),  # Bright star 5
    ]
    
    for x, y, brightness, size in constellation_stars:
        # Draw bright constellation stars
        for dx in range(-size*3, size*3+1):
            for dy in range(-size*3, size*3+1):
                if 0 <= x + dx < width and 0 <= y + dy < height:
                    distance = np.sqrt(dx*dx + dy*dy)
                    if distance <= size*3:
                        intensity = int(brightness * np.exp(-distance*distance / (2*size*size)))
                        for c in range(3):
                            image[y+dy, x+dx, c] = min(255, image[y+dy, x+dx, c] + intensity)
    
    return image, stars, constellation_stars

def main():
    """Create a test image."""
    print("🌟 Dinner Plate - Synthetic Test Image Creator")
    print("=" * 50)
    
    # Create synthetic star field
    image, stars, constellation_stars = create_synthetic_star_field(1024, 768, 300)
    
    # Save the image to Processing folder
    output_path = "Processing/synthetic_star_field.jpg"
    success = cv2.imwrite(output_path, image)
    
    if success:
        file_size = len(image.tobytes()) / (1024 * 1024)  # Approximate size
        print(f"✅ Synthetic star field created: {output_path}")
        print(f"📐 Image size: {image.shape[1]} x {image.shape[0]} pixels")
        print(f"⭐ Total stars: {len(stars)}")
        print(f"🌟 Constellation stars: {len(constellation_stars)}")
        print(f"📁 Approximate size: {file_size:.2f} MB")
        
        print(f"\n🎯 Now you can test the constellation annotator:")
        print(f"   python constellation_annotator.py {output_path} synthetic_annotated.jpg --verbose")
        
        return True
    else:
        print("❌ Failed to save synthetic image")
        return False

if __name__ == "__main__":
    main() 