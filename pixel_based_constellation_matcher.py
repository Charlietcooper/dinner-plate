#!/usr/bin/env python3
"""
Pixel-Based Constellation Matcher
Works with detected stars in pixel coordinates using pattern matching
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import json
import time
from typing import List, Tuple, Dict, Optional
import logging
from tqdm import tqdm
import math

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PixelBasedConstellationMatcher:
    """Constellation matcher that works with pixel coordinates using pattern matching."""
    
    def __init__(self):
        self.constellation_patterns = self._create_constellation_patterns()
        self.detected_stars = None
        self.fov_info = None
        
    def _create_constellation_patterns(self) -> Dict:
        """Create constellation patterns as relative distances and angles."""
        return {
            "Crux": {
                "name": "Southern Cross",
                "pattern": [
                    # Acrux to Mimosa
                    {"distance_ratio": 1.0, "angle": 0},
                    # Mimosa to Gacrux  
                    {"distance_ratio": 0.8, "angle": 90},
                    # Gacrux to Delta Crucis
                    {"distance_ratio": 0.7, "angle": 180},
                    # Delta Crucis to Acrux
                    {"distance_ratio": 0.9, "angle": 270}
                ],
                "min_stars": 4,
                "description": "Southern Cross - distinctive cross shape"
            },
            "Orion": {
                "name": "Orion",
                "pattern": [
                    # Belt stars (3 in a line)
                    {"distance_ratio": 1.0, "angle": 0},
                    {"distance_ratio": 1.0, "angle": 0},
                    # Shoulder to belt
                    {"distance_ratio": 1.2, "angle": 90},
                    # Belt to feet
                    {"distance_ratio": 1.5, "angle": 270}
                ],
                "min_stars": 5,
                "description": "Orion - hunter with distinctive belt"
            },
            "Ursa Major": {
                "name": "Big Dipper",
                "pattern": [
                    # Dipper handle
                    {"distance_ratio": 1.0, "angle": 0},
                    {"distance_ratio": 1.0, "angle": 0},
                    # Dipper bowl
                    {"distance_ratio": 0.8, "angle": 90},
                    {"distance_ratio": 0.8, "angle": 0},
                    {"distance_ratio": 0.8, "angle": 270}
                ],
                "min_stars": 5,
                "description": "Big Dipper - distinctive dipper shape"
            },
            "Cassiopeia": {
                "name": "Cassiopeia",
                "pattern": [
                    # W shape
                    {"distance_ratio": 1.0, "angle": 0},
                    {"distance_ratio": 0.8, "angle": 45},
                    {"distance_ratio": 0.8, "angle": 315},
                    {"distance_ratio": 1.0, "angle": 0}
                ],
                "min_stars": 5,
                "description": "Cassiopeia - W or M shape"
            },
            "Leo": {
                "name": "Leo",
                "pattern": [
                    # Sickle shape
                    {"distance_ratio": 1.0, "angle": 0},
                    {"distance_ratio": 0.7, "angle": 45},
                    {"distance_ratio": 0.7, "angle": 90},
                    {"distance_ratio": 0.7, "angle": 135},
                    {"distance_ratio": 0.7, "angle": 180}
                ],
                "min_stars": 5,
                "description": "Leo - sickle shape"
            }
        }
    
    def load_processed_data(self):
        """Load the processed star field data."""
        logger.info("📊 Loading processed data...")
        
        try:
            # Load improved star data
            with open("Processing/improved_detected_stars.json", 'r') as f:
                self.detected_stars = json.load(f)
            
            # Load improved FOV info
            with open("Processing/improved_fov_estimation.json", 'r') as f:
                self.fov_info = json.load(f)
            
            logger.info(f"   Loaded {len(self.detected_stars)} stars")
            logger.info(f"   FOV estimate: {self.fov_info.get('fov_estimate', 'unknown')}")
            logger.info(f"   Star density: {self.fov_info.get('star_density', 0):.6f} stars/pixel²")
            
        except FileNotFoundError as e:
            logger.error(f"❌ Failed to load data: {e}")
            raise
    
    def find_constellation_patterns(self) -> List[Dict]:
        """Find constellation patterns in the detected stars."""
        logger.info("🔍 Searching for constellation patterns...")
        
        if not self.detected_stars:
            raise ValueError("No detected stars available")
        
        matches = []
        
        # Sort stars by brightness for better matching
        bright_stars = sorted(self.detected_stars, key=lambda s: s.get('brightness', 0), reverse=True)
        
        # Take top 100 brightest stars for pattern matching
        candidate_stars = bright_stars[:100]
        
        for const_name, const_data in self.constellation_patterns.items():
            logger.info(f"   Searching for {const_name}...")
            
            pattern_matches = self._find_pattern_matches(candidate_stars, const_data)
            
            for match in pattern_matches:
                matches.append({
                    "constellation": const_name,
                    "constellation_name": const_data["name"],
                    "description": const_data["description"],
                    "confidence": match["confidence"],
                    "matched_stars": match["stars"],
                    "pattern_points": match["pattern_points"]
                })
        
        # Sort by confidence
        matches.sort(key=lambda m: m["confidence"], reverse=True)
        
        logger.info(f"   Found {len(matches)} potential constellation matches")
        return matches
    
    def _find_pattern_matches(self, stars: List[Dict], const_data: Dict) -> List[Dict]:
        """Find matches for a specific constellation pattern."""
        matches = []
        pattern = const_data["pattern"]
        min_stars = const_data["min_stars"]
        
        # Try different starting stars
        for start_star in stars[:20]:  # Limit to top 20 brightest
            pattern_matches = self._try_pattern_from_star(start_star, stars, pattern, min_stars)
            matches.extend(pattern_matches)
        
        return matches
    
    def _try_pattern_from_star(self, start_star: Dict, all_stars: List[Dict], 
                              pattern: List[Dict], min_stars: int) -> List[Dict]:
        """Try to match a pattern starting from a specific star."""
        matches = []
        
        # Get all possible second stars within reasonable distance
        start_x, start_y = start_star["x"], start_star["y"]
        
        for second_star in all_stars:
            if second_star == start_star:
                continue
                
            # Calculate base distance and angle
            dx = second_star["x"] - start_x
            dy = second_star["y"] - start_y
            base_distance = math.sqrt(dx*dx + dy*dy)
            base_angle = math.degrees(math.atan2(dy, dx))
            
            if base_distance < 50:  # Too close
                continue
            if base_distance > 500:  # Too far
                continue
            
            # Try to match the pattern
            pattern_stars = [start_star, second_star]
            pattern_points = [(start_x, start_y), (second_star["x"], second_star["y"])]
            
            success = True
            total_score = 0
            
            for i, pattern_step in enumerate(pattern[1:], 1):  # Skip first step
                expected_distance = base_distance * pattern_step["distance_ratio"]
                expected_angle = base_angle + pattern_step["angle"]
                
                # Find best matching star
                best_star = None
                best_score = 0
                
                for candidate_star in all_stars:
                    if candidate_star in pattern_stars:
                        continue
                    
                    # Calculate actual distance and angle from previous star
                    prev_star = pattern_stars[-1]
                    dx = candidate_star["x"] - prev_star["x"]
                    dy = candidate_star["y"] - prev_star["y"]
                    actual_distance = math.sqrt(dx*dx + dy*dy)
                    actual_angle = math.degrees(math.atan2(dy, dx))
                    
                    # Calculate score based on distance and angle match
                    distance_error = abs(actual_distance - expected_distance) / expected_distance
                    angle_error = abs(actual_angle - expected_angle)
                    if angle_error > 180:
                        angle_error = 360 - angle_error
                    angle_error = angle_error / 180.0
                    
                    score = 1.0 / (1.0 + distance_error + angle_error)
                    
                    if score > best_score and distance_error < 0.5 and angle_error < 0.3:
                        best_score = score
                        best_star = candidate_star
                
                if best_star:
                    pattern_stars.append(best_star)
                    pattern_points.append((best_star["x"], best_star["y"]))
                    total_score += best_score
                else:
                    success = False
                    break
            
            if success and len(pattern_stars) >= min_stars:
                confidence = total_score / len(pattern)
                if confidence > 0.3:  # Reasonable threshold
                    matches.append({
                        "confidence": confidence,
                        "stars": pattern_stars,
                        "pattern_points": pattern_points
                    })
        
        return matches
    
    def create_annotated_image(self, matches: List[Dict], output_path: str = "Output/pixel_based_constellations.jpg"):
        """Create annotated image with found constellation patterns."""
        logger.info("🎨 Creating annotated image...")
        
        # Load original image
        original_image = cv2.imread("Input/test-1.jpg")
        if original_image is None:
            logger.error("❌ Could not load original image")
            return
        
        # Create annotation overlay
        annotated_image = original_image.copy()
        
        # Draw matched constellations
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for i, match in enumerate(matches[:5]):  # Limit to top 5 matches
            color = colors[i % len(colors)]
            self._draw_constellation_pattern(annotated_image, match, color)
        
        # Add information overlay
        annotated_image = self._add_info_overlay(annotated_image, matches)
        
        # Save result
        cv2.imwrite(output_path, annotated_image)
        logger.info(f"   ✅ Saved annotated image: {output_path}")
    
    def _draw_constellation_pattern(self, image: np.ndarray, match: Dict, color: Tuple[int, int, int]):
        """Draw a constellation pattern on the image."""
        pattern_points = match["pattern_points"]
        
        # Draw lines connecting the pattern
        for i in range(len(pattern_points) - 1):
            pt1 = tuple(map(int, pattern_points[i]))
            pt2 = tuple(map(int, pattern_points[i + 1]))
            cv2.line(image, pt1, pt2, color, 3)
        
        # Draw stars
        for point in pattern_points:
            cv2.circle(image, tuple(map(int, point)), 8, color, -1)
            cv2.circle(image, tuple(map(int, point)), 10, (255, 255, 255), 2)
        
        # Add constellation name
        if pattern_points:
            text_pos = tuple(map(int, pattern_points[0]))
            cv2.putText(image, match["constellation_name"], 
                       (text_pos[0] + 15, text_pos[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    
    def _add_info_overlay(self, image: np.ndarray, matches: List[Dict]) -> np.ndarray:
        """Add information overlay to the image."""
        # Convert to PIL for better text rendering
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 36)
            small_font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Add title
        title = "Pixel-Based Constellation Pattern Matching"
        draw.text((20, 20), title, fill=(255, 255, 255), font=font)
        
        # Add processing information
        y_offset = 80
        info_lines = [
            f"Stars analyzed: {len(self.detected_stars)}",
            f"Patterns found: {len(matches)}",
            f"FOV estimate: {self.fov_info.get('fov_estimate', 'unknown')}",
            f"Star density: {self.fov_info.get('star_density', 0):.6f} stars/pixel²"
        ]
        
        for line in info_lines:
            draw.text((20, y_offset), line, fill=(255, 255, 255), font=small_font)
            y_offset += 35
        
        # Add constellation details
        y_offset += 20
        for i, match in enumerate(matches[:5]):
            const_name = match["constellation_name"]
            confidence = match["confidence"]
            num_stars = len(match["matched_stars"])
            
            line = f"• {const_name}: {confidence:.2f} ({num_stars} stars)"
            draw.text((20, y_offset), line, fill=(0, 255, 255), font=small_font)
            y_offset += 30
        
        # Add method explanation
        y_offset += 20
        draw.text((20, y_offset), "Method: Pattern matching in pixel space", 
                 fill=(255, 165, 0), font=small_font)
        y_offset += 25
        draw.text((20, y_offset), "No WCS coordinates required", 
                 fill=(255, 165, 0), font=small_font)
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def main():
    """Test pixel-based constellation matching."""
    print("🔍 Pixel-Based Constellation Pattern Matcher")
    print("=" * 60)
    print("Using pattern matching in pixel coordinates")
    print("=" * 60)
    
    start_time = time.time()
    
    # Create matcher
    matcher = PixelBasedConstellationMatcher()
    
    try:
        # Load processed data
        matcher.load_processed_data()
        
        # Find constellation patterns
        matches = matcher.find_constellation_patterns()
        
        # Create annotated image
        matcher.create_annotated_image(matches)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ Pattern matching complete!")
        print(f"   Found {len(matches)} potential constellation patterns")
        
        if matches:
            print("   Top matches:")
            for i, match in enumerate(matches[:5]):
                print(f"     {i+1}. {match['constellation_name']}: {match['confidence']:.2f} confidence")
        else:
            print("   No clear constellation patterns found")
            print("   This could indicate:")
            print("     - Image shows a different sky region")
            print("     - Stars are too dense for pattern recognition")
            print("     - Need more sophisticated pattern matching")
        
        print(f"   Total processing time: {elapsed_time:.1f} seconds")
        print(f"   Check Output/pixel_based_constellations.jpg for results")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        raise

if __name__ == "__main__":
    main() 