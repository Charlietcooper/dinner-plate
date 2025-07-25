#!/usr/bin/env python3
"""
Resize image for better plate solving performance
"""

import cv2
import os

def resize_image(input_path, output_path, max_dimension=2048):
    """Resize image while maintaining aspect ratio."""
    
    print(f"📏 Resizing image: {input_path}")
    print(f"🎯 Target max dimension: {max_dimension} pixels")
    
    # Load image
    image = cv2.imread(input_path)
    if image is None:
        print("❌ Could not load image")
        return False
    
    height, width = image.shape[:2]
    print(f"📐 Original size: {width} x {height}")
    
    # Calculate new dimensions
    if width > height:
        new_width = max_dimension
        new_height = int(height * max_dimension / width)
    else:
        new_height = max_dimension
        new_width = int(width * max_dimension / height)
    
    print(f"📐 New size: {new_width} x {new_height}")
    
    # Resize image
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Save resized image
    success = cv2.imwrite(output_path, resized)
    
    if success:
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"✅ Resized image saved: {output_path}")
        print(f"📁 New file size: {file_size:.2f} MB")
        return True
    else:
        print("❌ Failed to save resized image")
        return False

def main():
    """Main function."""
    input_path = "/Users/charlie/Projects/Astro-Plate-Solve/test_images/test-1.jpg"
    output_path = "test-1_resized.jpg"
    
    print("🌟 Dinner Plate - Image Resizing")
    print("=" * 40)
    
    success = resize_image(input_path, output_path, max_dimension=2048)
    
    if success:
        print("\n" + "=" * 40)
        print("✅ Resizing complete!")
        print(f"📸 Try plate solving with: {output_path}")
    else:
        print("\n❌ Resizing failed!")

if __name__ == "__main__":
    main() 