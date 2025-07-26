#!/usr/bin/env python3
"""
Debug Galactic Center Star Identifier
To investigate why all stars are being placed in the same location
"""

import cv2
import numpy as np
import json
import logging
from typing import List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_star_identification():
    """Debug the star identification process."""
    
    # Load star data
    with open("Processing/improved_detected_stars.json", 'r') as f:
        detected_stars = json.load(f)
    
    print(f"Loaded {len(detected_stars)} stars")
    
    # Sort by brightness
    bright_stars = sorted(detected_stars, key=lambda s: s.get('brightness', 0), reverse=True)
    top_20 = bright_stars[:20]
    
    print("\nTop 20 brightest stars:")
    for i, star in enumerate(top_20):
        print(f"  {i+1}. x={star['x']}, y={star['y']}, brightness={star['brightness']:.1f}")
    
    # Test brightness matching for a specific star
    test_star_data = {
        "name": "Antares",
        "magnitude": 1.06
    }
    
    print(f"\nTesting brightness matching for {test_star_data['name']} (mag {test_star_data['magnitude']}):")
    
    # Convert magnitude to expected brightness
    expected_magnitude = test_star_data["magnitude"]
    expected_brightness = 255 * (1.0 / (1.0 + abs(expected_magnitude)))
    print(f"Expected brightness: {expected_brightness:.1f}")
    
    # Find matches
    matches = []
    for star in bright_stars[:100]:
        actual_brightness = star.get("brightness", 0)
        brightness_error = abs(actual_brightness - expected_brightness) / 255
        brightness_score = 1.0 / (1.0 + brightness_error)
        
        if brightness_score > 0.2:
            matches.append({
                **star,
                "brightness_score": brightness_score
            })
    
    # Sort by brightness score
    matches.sort(key=lambda m: m["brightness_score"], reverse=True)
    
    print(f"\nFound {len(matches)} potential matches:")
    for i, match in enumerate(matches[:10]):
        print(f"  {i+1}. x={match['x']}, y={match['y']}, brightness={match['brightness']:.1f}, score={match['brightness_score']:.3f}")
    
    # Check if any stars are being reused
    print(f"\nChecking for duplicate coordinates in top matches:")
    coordinates = [(m['x'], m['y']) for m in matches[:10]]
    unique_coords = set(coordinates)
    print(f"  Total matches: {len(coordinates)}")
    print(f"  Unique coordinates: {len(unique_coords)}")
    
    if len(coordinates) != len(unique_coords):
        print("  ⚠️  DUPLICATE COORDINATES FOUND!")
        # Find duplicates
        seen = set()
        duplicates = []
        for coord in coordinates:
            if coord in seen:
                duplicates.append(coord)
            else:
                seen.add(coord)
        print(f"  Duplicate coordinates: {duplicates}")
    
    # Test relationship validation
    print(f"\nTesting relationship validation:")
    
    # Simulate some identified stars
    identified_stars = [
        {
            "database_star": "Antares",
            "detected_star": matches[0] if matches else {"x": 100, "y": 100, "brightness": 200}
        }
    ]
    
    # Test relationship validation for a second star
    test_relationships = {
        "Antares": {"distance": 8.2, "angle": 180}
    }
    
    for i, match in enumerate(matches[:5]):
        print(f"\nTesting relationship for match {i+1} (x={match['x']}, y={match['y']}):")
        
        # Calculate relationship with first identified star
        star1 = identified_stars[0]["detected_star"]
        dx = match["x"] - star1["x"]
        dy = match["y"] - star1["y"]
        
        # Convert to angular distance
        pixel_to_degree = 0.001
        distance = (dx*dx + dy*dy)**0.5 * pixel_to_degree
        angle = np.degrees(np.arctan2(dy, dx))
        if angle < 0:
            angle += 360
        
        print(f"  Distance: {distance:.2f} degrees")
        print(f"  Angle: {angle:.1f} degrees")
        
        # Compare with expected
        expected_rel = test_relationships["Antares"]
        distance_error = abs(distance - expected_rel["distance"]) / expected_rel["distance"]
        angle_error = abs(angle - expected_rel["angle"])
        if angle_error > 180:
            angle_error = 360 - angle_error
        angle_error = angle_error / 180.0
        
        print(f"  Distance error: {distance_error:.3f}")
        print(f"  Angle error: {angle_error:.3f}")
        
        if distance_error < 0.8 and angle_error < 0.5:
            rel_score = 1.0 / (1.0 + distance_error + angle_error)
            print(f"  Relationship score: {rel_score:.3f}")
        else:
            print(f"  Relationship score: 0 (outside tolerances)")

if __name__ == "__main__":
    debug_star_identification() 