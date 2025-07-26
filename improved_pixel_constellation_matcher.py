#!/usr/bin/env python3
"""
Improved Pixel-Based Constellation Matcher
Better filtering and more realistic confidence scores
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

class ImprovedPixelConstellationMatcher:
    """Improved constellation matcher with better filtering."""
    
    def __init__(self):
        self.constellation_patterns = self._create_constellation_patterns()
        self.detected_stars = None
        self.fov_info = None
        
    def _create_constellation_patterns(self) -> Dict:
        """Create more realistic constellation patterns."""
        return {
            "Crux": {
                "name": "Southern Cross",
                "pattern": [
                    {"distance_ratio": 1.0, "angle": 0},      # Acrux to Mimosa
                    {"distance_ratio": 0.8, "angle": 90},     # Mimosa to Gacrux
                    {"distance_ratio": 0.7, "angle": 180},    # Gacrux to Delta Crucis
                    {"distance_ratio": 0.9, "angle": 270}     # Delta Crucis to Acrux
                ],
                "min_stars": 4,
                "min_confidence": 0.6,
                "description": "Southern Cross - distinctive cross shape"
            },
            "Orion": {
                "name": "Orion",
                "pattern": [
                    {"distance_ratio": 1.0, "angle": 0},      # Belt star 1 to 2
                    {"distance_ratio": 1.0, "angle": 0},      # Belt star 2 to 3
                    {"distance_ratio": 1.2, "angle": 90},     # Shoulder to belt
                    {"distance_ratio": 1.5, "angle": 270}     # Belt to feet
                ],
                "min_stars": 5,
                "min_confidence": 0.5,
                "description": "Orion - hunter with distinctive belt"
            },
            "Ursa Major": {
                "name": "Big Dipper",
                "pattern": [
                    {"distance_ratio": 1.0, "angle": 0},      # Handle star 1 to 2
                    {"distance_ratio": 1.0, "angle": 0},      # Handle star 2 to 3
                    {"distance_ratio": 0.8, "angle": 90},     # Handle to bowl
                    {"distance_ratio": 0.8, "angle": 0},      # Bowl star 1 to 2
                    {"distance_ratio": 0.8, "angle": 270}     # Bowl star 2 to 3
                ],
                "min_stars": 5,
                "min_confidence": 0.5,
                "description": "Big Dipper - distinctive dipper shape"
            },
            "Cassiopeia": {
                "name": "Cassiopeia",
                "pattern": [
                    {"distance_ratio": 1.0, "angle": 0},      # W point 1 to 2
                    {"distance_ratio": 0.8, "angle": 45},     # W point 2 to 3
                    {"distance_ratio": 0.8, "angle": 315},    # W point 3 to 4
                    {"distance_ratio": 1.0, "angle": 0}       # W point 4 to 5
                ],
                "min_stars": 5,
                "min_confidence": 0.5,
                "description": "Cassiopeia - W or M shape"
            }
        }
    
    def load_processed_data(self):
        """Load the processed star field data."""
        logger.info("📊 Loading processed data...")
        
        try:
            with open("Processing/improved_detected_stars.json", 'r') as f:
                self.detected_stars = json.load(f)
            
            with open("Processing/improved_fov_estimation.json", 'r') as f:
                self.fov_info = json.load(f)
            
            logger.info(f"   Loaded {len(self.detected_stars)} stars")
            logger.info(f"   FOV estimate: {self.fov_info.get('fov_estimate', 'unknown')}")
            
        except FileNotFoundError as e:
            logger.error(f"❌ Failed to load data: {e}")
            raise
    
    def find_constellation_patterns(self) -> List[Dict]:
        """Find constellation patterns with improved filtering."""
        logger.info("🔍 Searching for constellation patterns...")
        
        if not self.detected_stars:
            raise ValueError("No detected stars available")
        
        matches = []
        
        # Sort stars by brightness and take top candidates
        bright_stars = sorted(self.detected_stars, key=lambda s: s.get('brightness', 0), reverse=True)
        candidate_stars = bright_stars[:50]  # Reduced from 100
        
        for const_name, const_data in self.constellation_patterns.items():
            logger.info(f"   Searching for {const_name}...")
            
            pattern_matches = self._find_pattern_matches(candidate_stars, const_data)
            
            # Filter by minimum confidence
            filtered_matches = [m for m in pattern_matches if m["confidence"] >= const_data["min_confidence"]]
            
            # Take only the best match for each constellation
            if filtered_matches:
                best_match = max(filtered_matches, key=lambda m: m["confidence"])
                matches.append({
                    "constellation": const_name,
                    "constellation_name": const_data["name"],
                    "description": const_data["description"],
                    "confidence": best_match["confidence"],
                    "matched_stars": best_match["stars"],
                    "pattern_points": best_match["pattern_points"]
                })
        
        # Sort by confidence
        matches.sort(key=lambda m: m["confidence"], reverse=True)
        
        logger.info(f"   Found {len(matches)} high-confidence constellation matches")
        return matches
    
    def _find_pattern_matches(self, stars: List[Dict], const_data: Dict) -> List[Dict]:
        """Find matches for a specific constellation pattern."""
        matches = []
        pattern = const_data["pattern"]
        min_stars = const_data["min_stars"]
        
        # Try different starting stars
        for start_star in stars[:10]:  # Reduced from 20
            pattern_matches = self._try_pattern_from_star(start_star, stars, pattern, min_stars)
            matches.extend(pattern_matches)
        
        return matches
    
    def _try_pattern_from_star(self, start_star: Dict, all_stars: List[Dict], 
                              pattern: List[Dict], min_stars: int) -> List[Dict]:
        """Try to match a pattern starting from a specific star."""
        matches = []
        
        start_x, start_y = start_star["x"], start_star["y"]
        
        for second_star in all_stars:
            if second_star == start_star:
                continue
                
            # Calculate base distance and angle
            dx = second_star["x"] - start_x
            dy = second_star["y"] - start_y
            base_distance = math.sqrt(dx*dx + dy*dy)
            base_angle = math.degrees(math.atan2(dy, dx))
            
            # More restrictive distance limits
            if base_distance < 100:  # Increased minimum
                continue
            if base_distance > 300:  # Reduced maximum
                continue
            
            # Try to match the pattern
            pattern_stars = [start_star, second_star]
            pattern_points = [(start_x, start_y), (second_star["x"], second_star["y"])]
            
            success = True
            total_score = 0
            
            for i, pattern_step in enumerate(pattern[1:], 1):
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
                    
                    # More restrictive matching criteria
                    if distance_error < 0.3 and angle_error < 0.2:  # Tighter tolerances
                        score = 1.0 / (1.0 + distance_error + angle_error)
                        if score > best_score:
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
                if confidence > 0.4:  # Higher threshold
                    matches.append({
                        "confidence": confidence,
                        "stars": pattern_stars,
                        "pattern_points": pattern_points
                    })
        
        return matches
    
    def create_annotated_image(self, matches: List[Dict], output_path: str = "Output/improved_constellations.jpg"):
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
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
        
        for i, match in enumerate(matches):
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
            cv2.line(image, pt1, pt2, color, 4)
        
        # Draw stars
        for point in pattern_points:
            cv2.circle(image, tuple(map(int, point)), 10, color, -1)
            cv2.circle(image, tuple(map(int, point)), 12, (255, 255, 255), 2)
        
        # Add constellation name and confidence
        if pattern_points:
            text_pos = tuple(map(int, pattern_points[0]))
            label = f"{match['constellation_name']} ({match['confidence']:.2f})"
            cv2.putText(image, label, 
                       (text_pos[0] + 15, text_pos[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
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
        title = "Improved Constellation Pattern Matching"
        draw.text((20, 20), title, fill=(255, 255, 255), font=font)
        
        # Add processing information
        y_offset = 80
        info_lines = [
            f"Stars analyzed: {len(self.detected_stars)}",
            f"High-confidence matches: {len(matches)}",
            f"FOV estimate: {self.fov_info.get('fov_estimate', 'unknown')}",
            f"Method: Pixel-based pattern matching"
        ]
        
        for line in info_lines:
            draw.text((20, y_offset), line, fill=(255, 255, 255), font=small_font)
            y_offset += 35
        
        # Add constellation details
        y_offset += 20
        for i, match in enumerate(matches):
            const_name = match["constellation_name"]
            confidence = match["confidence"]
            num_stars = len(match["matched_stars"])
            
            line = f"• {const_name}: {confidence:.2f} confidence ({num_stars} stars)"
            draw.text((20, y_offset), line, fill=(0, 255, 255), font=small_font)
            y_offset += 30
        
        # Add explanation
        y_offset += 20
        draw.text((20, y_offset), "Filtered for high-confidence matches only", 
                 fill=(255, 165, 0), font=small_font)
        y_offset += 25
        draw.text((20, y_offset), "Tighter matching criteria applied", 
                 fill=(255, 165, 0), font=small_font)
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def main():
    """Test improved pixel-based constellation matching."""
    print("🔍 Improved Pixel-Based Constellation Matcher")
    print("=" * 60)
    print("Better filtering and realistic confidence scores")
    print("=" * 60)
    
    start_time = time.time()
    
    # Create matcher
    matcher = ImprovedPixelConstellationMatcher()
    
    try:
        # Load processed data
        matcher.load_processed_data()
        
        # Find constellation patterns
        matches = matcher.find_constellation_patterns()
        
        # Create annotated image
        matcher.create_annotated_image(matches)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ Improved pattern matching complete!")
        print(f"   Found {len(matches)} high-confidence constellation matches")
        
        if matches:
            print("   Matched constellations:")
            for i, match in enumerate(matches):
                print(f"     {i+1}. {match['constellation_name']}: {match['confidence']:.2f} confidence")
        else:
            print("   No high-confidence constellation patterns found")
            print("   This could indicate:")
            print("     - Image shows a different sky region")
            print("     - Need more sophisticated pattern recognition")
            print("     - Image may be too dense for simple pattern matching")
        
        print(f"   Total processing time: {elapsed_time:.1f} seconds")
        print(f"   Check Output/improved_constellations.jpg for results")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        raise

if __name__ == "__main__":
    main() 