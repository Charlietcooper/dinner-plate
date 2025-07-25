#!/usr/bin/env python3
"""
Test image analysis for Dinner Plate constellation annotator
"""

import cv2
import numpy as np
from PIL import Image
import os

def analyze_image(image_path):
    """Analyze an image to check if it's suitable for plate solving."""
    
    print(f"🔍 Analyzing image: {image_path}")
    print("=" * 50)
    
    # Check if file exists
    if not os.path.exists(image_path):
        print("❌ Image file not found!")
        return False
    
    # Get file info
    file_size = os.path.getsize(image_path) / (1024 * 1024)  # MB
    print(f"📁 File size: {file_size:.2f} MB")
    
    # Load image with OpenCV
    try:
        image = cv2.imread(image_path)
        if image is None:
            print("❌ Could not load image with OpenCV")
            return False
        
        height, width, channels = image.shape
        print(f"📐 Image dimensions: {width} x {height} pixels")
        print(f"🎨 Color channels: {channels}")
        
        # Check image statistics
        mean_brightness = np.mean(image)
        std_brightness = np.std(image)
        print(f"💡 Mean brightness: {mean_brightness:.2f}")
        print(f"📊 Brightness std dev: {std_brightness:.2f}")
        
        # Convert to grayscale for star detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Simple star detection (bright spots)
        # Threshold to find bright pixels
        threshold = np.mean(gray) + 2 * np.std(gray)
        bright_pixels = np.sum(gray > threshold)
        total_pixels = gray.shape[0] * gray.shape[1]
        bright_ratio = bright_pixels / total_pixels * 100
        
        print(f"⭐ Bright pixels (> threshold): {bright_pixels:,}")
        print(f"📊 Bright pixel ratio: {bright_ratio:.3f}%")
        
        # Check if image looks like it has stars
        if bright_ratio < 0.1:
            print("⚠️  Very few bright pixels - might not be a star field")
        elif bright_ratio > 10:
            print("⚠️  Very bright image - might be overexposed")
        else:
            print("✅ Bright pixel ratio looks reasonable for a star field")
        
        # Try to find star-like objects
        # Use simple blob detection
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 50
        params.maxThreshold = 200
        params.filterByArea = True
        params.minArea = 10
        params.maxArea = 1000
        params.filterByCircularity = True
        params.minCircularity = 0.3
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray)
        
        print(f"🔍 Detected {len(keypoints)} potential star-like objects")
        
        if len(keypoints) < 10:
            print("⚠️  Very few star-like objects detected")
            print("   This might not be a suitable astronomical image")
        elif len(keypoints) < 50:
            print("⚠️  Few star-like objects - plate solving might be difficult")
        else:
            print("✅ Good number of star-like objects detected")
        
        # Save a preview with detected objects
        preview = cv2.drawKeypoints(image, keypoints, np.array([]), 
                                  (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        preview_path = "Processing/image_analysis_preview.jpg"
        cv2.imwrite(preview_path, preview)
        print(f"📸 Preview saved as: {preview_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing image: {e}")
        return False

def main():
    """Main function."""
    image_path = "/Users/charlie/Projects/Astro-Plate-Solve/test_images/test-1.jpg"
    
    print("🌟 Dinner Plate - Image Analysis")
    print("=" * 50)
    
    success = analyze_image(image_path)
    
    if success:
        print("\n" + "=" * 50)
        print("📋 Analysis Complete!")
        print("\n💡 Recommendations:")
        print("1. Check the preview image to see detected objects")
        print("2. If few objects detected, the image might not be suitable")
        print("3. Try with a different astronomical image")
        print("4. Consider resizing the image if it's very large")
    else:
        print("\n❌ Analysis failed!")

if __name__ == "__main__":
    main() 