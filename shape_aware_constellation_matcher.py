#!/usr/bin/env python3
"""
Shape-Aware Constellation Matcher
Uses proper constellation shapes from IAU data but adapts to pixel coordinates
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import json
import time
from typing import List, Tuple, Dict, Optional, Set
import logging
from tqdm import tqdm
import math

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ShapeAwareConstellationMatcher:
    """Constellation matcher that uses proper shapes but adapts to pixel coordinates."""
    
    def __init__(self):
        self.constellation_shapes = self._create_constellation_shapes()
        self.detected_stars = None
        self.fov_info = None
        
    def _create_constellation_shapes(self) -> Dict:
        """Create constellation shapes based on IAU data."""
        return {
            "Crux": {
                "name": "Southern Cross",
                "hemisphere": "southern",
                "shape_type": "cross",
                "expected_ratios": [1.0, 0.8, 0.7, 0.9],  # Relative distances
                "expected_angles": [0, 90, 180, 270],      # Relative angles
                "min_stars": 4,
                "min_confidence": 0.6,
                "description": "Southern Cross - distinctive cross shape"
            },
            "Carina": {
                "name": "Carina",
                "hemisphere": "southern", 
                "shape_type": "ship_keel",
                "expected_ratios": [1.0, 0.9, 0.8],
                "expected_angles": [0, 45, 90],
                "min_stars": 4,
                "min_confidence": 0.5,
                "description": "Carina - keel of the ship"
            },
            "Centaurus": {
                "name": "Centaurus",
                "hemisphere": "southern",
                "shape_type": "centaur",
                "expected_ratios": [1.0, 0.8, 0.7],
                "expected_angles": [0, 60, 120],
                "min_stars": 4,
                "min_confidence": 0.5,
                "description": "Centaurus - the centaur"
            },
            "Orion": {
                "name": "Orion",
                "hemisphere": "both",
                "shape_type": "hunter",
                "expected_ratios": [1.0, 1.0, 1.2, 1.5],  # Belt + shoulders + feet
                "expected_angles": [0, 0, 90, 270],       # Belt horizontal, shoulders/feet vertical
                "min_stars": 5,
                "min_confidence": 0.5,
                "description": "Orion - hunter with distinctive belt"
            },
            "Ursa Major": {
                "name": "Big Dipper",
                "hemisphere": "northern",
                "shape_type": "dipper",
                "expected_ratios": [1.0, 1.0, 0.8, 0.8, 0.8],  # Handle + bowl
                "expected_angles": [0, 0, 90, 0, 270],         # Handle straight, bowl curves
                "min_stars": 5,
                "min_confidence": 0.5,
                "description": "Big Dipper - distinctive dipper shape"
            },
            "Cassiopeia": {
                "name": "Cassiopeia",
                "hemisphere": "northern",
                "shape_type": "w_shape",
                "expected_ratios": [1.0, 0.8, 0.8, 1.0],
                "expected_angles": [0, 45, 315, 0],  # W or M shape
                "min_stars": 5,
                "min_confidence": 0.5,
                "description": "Cassiopeia - W or M shape"
            },
            "Leo": {
                "name": "Leo",
                "hemisphere": "both",
                "shape_type": "sickle",
                "expected_ratios": [1.0, 0.7, 0.7, 0.7],
                "expected_angles": [0, 45, 90, 135],  # Sickle curve
                "min_stars": 4,
                "min_confidence": 0.5,
                "description": "Leo - sickle shape"
            },
            "Virgo": {
                "name": "Virgo",
                "hemisphere": "both",
                "shape_type": "maiden",
                "expected_ratios": [1.0, 0.8, 0.7],
                "expected_angles": [0, 45, 90],
                "min_stars": 4,
                "min_confidence": 0.5,
                "description": "Virgo - the maiden"
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
        """Find constellation patterns using shape-aware matching."""
        logger.info("🔍 Searching for constellations with shape-aware matching...")
        
        if not self.detected_stars:
            raise ValueError("No detected stars available")
        
        # Sort stars by brightness
        bright_stars = sorted(self.detected_stars, key=lambda s: s.get('brightness', 0), reverse=True)
        candidate_stars = bright_stars[:100]  # Use top 100 brightest stars
        
        matches = []
        
        for const_name, const_data in self.constellation_shapes.items():
            logger.info(f"   Searching for {const_name}...")
            
            shape_matches = self._find_shape_matches(candidate_stars, const_data)
            
            # Filter by minimum confidence
            filtered_matches = [m for m in shape_matches if m["confidence"] >= const_data["min_confidence"]]
            
            for match in filtered_matches:
                matches.append({
                    "constellation": const_name,
                    "constellation_name": const_data["name"],
                    "hemisphere": const_data["hemisphere"],
                    "description": const_data["description"],
                    "confidence": match["confidence"],
                    "matched_stars": match["stars"],
                    "pattern_points": match["pattern_points"],
                    "center_x": np.mean([s["x"] for s in match["stars"]]),
                    "center_y": np.mean([s["y"] for s in match["stars"]]),
                    "shape_score": match["shape_score"]
                })
        
        # Apply spatial constraints
        constrained_matches = self._apply_spatial_constraints(matches)
        
        # Sort by confidence
        constrained_matches.sort(key=lambda m: m["confidence"], reverse=True)
        
        logger.info(f"   Found {len(constrained_matches)} shape-aware constellation matches")
        return constrained_matches
    
    def _find_shape_matches(self, stars: List[Dict], const_data: Dict) -> List[Dict]:
        """Find matches for a specific constellation shape."""
        matches = []
        expected_ratios = const_data["expected_ratios"]
        expected_angles = const_data["expected_angles"]
        min_stars = const_data["min_stars"]
        
        # Try different starting stars
        for start_star in stars[:20]:  # Try top 20 brightest as starting points
            shape_matches = self._try_shape_from_star(start_star, stars, expected_ratios, expected_angles, min_stars)
            matches.extend(shape_matches)
        
        return matches
    
    def _try_shape_from_star(self, start_star: Dict, all_stars: List[Dict], 
                            expected_ratios: List[float], expected_angles: List[float], 
                            min_stars: int) -> List[Dict]:
        """Try to match a constellation shape starting from a specific star."""
        matches = []
        
        start_x, start_y = start_star["x"], start_star["y"]
        
        # Try different second stars to establish the base direction
        for second_star in all_stars:
            if second_star == start_star:
                continue
                
            # Calculate base distance and angle
            dx = second_star["x"] - start_x
            dy = second_star["y"] - start_y
            base_distance = math.sqrt(dx*dx + dy*dy)
            base_angle = math.degrees(math.atan2(dy, dx))
            
            # Distance constraints
            if base_distance < 80:  # Too close
                continue
            if base_distance > 400:  # Too far
                continue
            
            # Try to build the constellation shape
            shape_stars = [start_star, second_star]
            shape_points = [(start_x, start_y), (second_star["x"], second_star["y"])]
            
            success = True
            total_score = 0
            shape_score = 0
            
            # Build the shape step by step
            for i, (expected_ratio, expected_angle) in enumerate(zip(expected_ratios[1:], expected_angles[1:]), 1):
                target_distance = base_distance * expected_ratio
                target_angle = base_angle + expected_angle
                
                # Find best matching star
                best_star = None
                best_score = 0
                
                for candidate_star in all_stars:
                    if candidate_star in shape_stars:
                        continue
                    
                    # Calculate distance and angle from previous star
                    prev_star = shape_stars[-1]
                    dx = candidate_star["x"] - prev_star["x"]
                    dy = candidate_star["y"] - prev_star["y"]
                    actual_distance = math.sqrt(dx*dx + dy*dy)
                    actual_angle = math.degrees(math.atan2(dy, dx))
                    
                    # Calculate shape matching score
                    distance_error = abs(actual_distance - target_distance) / target_distance
                    angle_error = abs(actual_angle - target_angle)
                    if angle_error > 180:
                        angle_error = 360 - angle_error
                    angle_error = angle_error / 180.0
                    
                    # Combined score
                    if distance_error < 0.4 and angle_error < 0.3:  # Reasonable tolerances
                        score = 1.0 / (1.0 + distance_error + angle_error)
                        if score > best_score:
                            best_score = score
                            best_star = candidate_star
                
                if best_star:
                    shape_stars.append(best_star)
                    shape_points.append((best_star["x"], best_star["y"]))
                    total_score += best_score
                    shape_score += best_score
                else:
                    success = False
                    break
            
            if success and len(shape_stars) >= min_stars:
                # Calculate final confidence
                confidence = total_score / len(expected_ratios)
                
                # Add brightness bonus
                brightness_bonus = sum(s.get("brightness", 0) for s in shape_stars) / (len(shape_stars) * 255)
                confidence = 0.7 * confidence + 0.3 * brightness_bonus
                
                if confidence > 0.3:  # Minimum threshold
                    matches.append({
                        "confidence": confidence,
                        "stars": shape_stars,
                        "pattern_points": shape_points,
                        "shape_score": shape_score / len(expected_ratios)
                    })
        
        return matches
    
    def _apply_spatial_constraints(self, matches: List[Dict]) -> List[Dict]:
        """Apply spatial constraints to filter matches."""
        logger.info("🌍 Applying spatial constraints...")
        
        if not matches:
            return []
        
        # Group by hemisphere
        northern_matches = [m for m in matches if m["hemisphere"] == "northern"]
        southern_matches = [m for m in matches if m["hemisphere"] == "southern"]
        both_matches = [m for m in matches if m["hemisphere"] == "both"]
        
        logger.info(f"   Northern: {len(northern_matches)}, Southern: {len(southern_matches)}, Both: {len(both_matches)}")
        
        # Rule 1: Hemisphere consistency
        if northern_matches and southern_matches:
            logger.info("   ❌ Detected both northern and southern constellations - applying hemisphere constraint")
            
            northern_score = sum(m["confidence"] for m in northern_matches)
            southern_score = sum(m["confidence"] for m in southern_matches)
            
            if northern_score > southern_score:
                logger.info(f"   ✅ Keeping northern hemisphere (score: {northern_score:.2f})")
                matches = northern_matches + both_matches
            else:
                logger.info(f"   ✅ Keeping southern hemisphere (score: {southern_score:.2f})")
                matches = southern_matches + both_matches
        
        # Rule 2: Remove duplicates (keep highest confidence)
        unique_matches = {}
        for match in matches:
            const_name = match["constellation"]
            if const_name not in unique_matches or match["confidence"] > unique_matches[const_name]["confidence"]:
                unique_matches[const_name] = match
        
        matches = list(unique_matches.values())
        
        # Rule 3: Shape quality validation
        valid_matches = []
        for match in matches:
            shape_score = match.get("shape_score", 0)
            if shape_score > 0.4:  # Minimum shape quality
                valid_matches.append(match)
                logger.info(f"   ✅ Added {match['constellation_name']} (shape score: {shape_score:.2f})")
            else:
                logger.info(f"   ❌ Rejected {match['constellation_name']} (poor shape match: {shape_score:.2f})")
        
        # Rule 4: Spatial separation
        final_matches = []
        for match in valid_matches:
            is_valid = True
            for final_match in final_matches:
                distance = self._calculate_constellation_distance(match, final_match)
                if distance < 250:  # Minimum separation
                    is_valid = False
                    logger.info(f"   ❌ Rejected {match['constellation_name']} (too close to {final_match['constellation_name']})")
                    break
            
            if is_valid:
                final_matches.append(match)
                logger.info(f"   ✅ Added {match['constellation_name']} (spatially separated)")
        
        # Rule 5: FOV limits
        max_constellations = self._get_max_constellations_for_fov()
        if len(final_matches) > max_constellations:
            logger.info(f"   Limiting to {max_constellations} constellations based on FOV")
            final_matches = final_matches[:max_constellations]
        
        return final_matches
    
    def _calculate_constellation_distance(self, match1: Dict, match2: Dict) -> float:
        """Calculate distance between two constellation centers."""
        dx = match1["center_x"] - match2["center_x"]
        dy = match1["center_y"] - match2["center_y"]
        return math.sqrt(dx*dx + dy*dy)
    
    def _get_max_constellations_for_fov(self) -> int:
        """Get maximum number of constellations based on FOV."""
        fov_estimate = self.fov_info.get("fov_estimate", "unknown")
        
        if "narrow_field" in fov_estimate:
            return 2
        elif "medium_field" in fov_estimate:
            return 3
        else:
            return 4
    
    def create_annotated_image(self, matches: List[Dict], output_path: str = "Output/shape_aware_constellations.jpg"):
        """Create annotated image with shape-aware constellation patterns."""
        logger.info("🎨 Creating shape-aware annotated image...")
        
        # Load original image
        original_image = cv2.imread("Input/test-1.jpg")
        if original_image is None:
            logger.error("❌ Could not load original image")
            return
        
        # Create annotation overlay
        annotated_image = original_image.copy()
        
        # Draw matched constellations
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
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
        
        # Add constellation name, confidence, and shape score
        if pattern_points:
            text_pos = tuple(map(int, pattern_points[0]))
            hemisphere = match["hemisphere"]
            shape_score = match.get("shape_score", 0)
            label = f"{match['constellation_name']} ({match['confidence']:.2f}) [{hemisphere}]"
            cv2.putText(image, label, 
                       (text_pos[0] + 15, text_pos[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            
            # Add shape score
            shape_label = f"Shape: {shape_score:.2f}"
            cv2.putText(image, shape_label,
                       (text_pos[0] + 15, text_pos[1] + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
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
        title = "Shape-Aware Constellation Matching"
        draw.text((20, 20), title, fill=(255, 255, 255), font=font)
        
        # Add processing information
        y_offset = 80
        info_lines = [
            f"Stars analyzed: {len(self.detected_stars)}",
            f"Shape-valid matches: {len(matches)}",
            f"FOV estimate: {self.fov_info.get('fov_estimate', 'unknown')}",
            f"Method: IAU shape patterns with pixel adaptation"
        ]
        
        for line in info_lines:
            draw.text((20, y_offset), line, fill=(255, 255, 255), font=small_font)
            y_offset += 35
        
        # Add constellation details
        y_offset += 20
        for i, match in enumerate(matches):
            const_name = match["constellation_name"]
            confidence = match["confidence"]
            hemisphere = match["hemisphere"]
            shape_score = match.get("shape_score", 0)
            num_stars = len(match["matched_stars"])
            
            line = f"• {const_name}: {confidence:.2f} confidence ({hemisphere}, {num_stars} stars)"
            draw.text((20, y_offset), line, fill=(0, 255, 255), font=small_font)
            y_offset += 25
            line2 = f"  Shape score: {shape_score:.2f}"
            draw.text((20, y_offset), line2, fill=(255, 165, 0), font=small_font)
            y_offset += 30
        
        # Add constraint information
        y_offset += 20
        draw.text((20, y_offset), "Applied constraints:", 
                 fill=(255, 165, 0), font=small_font)
        y_offset += 25
        draw.text((20, y_offset), "• IAU constellation shapes", 
                 fill=(255, 165, 0), font=small_font)
        y_offset += 25
        draw.text((20, y_offset), "• Shape quality validation", 
                 fill=(255, 165, 0), font=small_font)
        y_offset += 25
        draw.text((20, y_offset), "• No forced pattern fitting", 
                 fill=(255, 165, 0), font=small_font)
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def main():
    """Test shape-aware constellation matching."""
    print("🔍 Shape-Aware Constellation Matcher")
    print("=" * 60)
    print("Using IAU shapes with pixel coordinate adaptation")
    print("=" * 60)
    
    start_time = time.time()
    
    # Create matcher
    matcher = ShapeAwareConstellationMatcher()
    
    try:
        # Load processed data
        matcher.load_processed_data()
        
        # Find constellation patterns
        matches = matcher.find_constellation_patterns()
        
        # Create annotated image
        matcher.create_annotated_image(matches)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ Shape-aware matching complete!")
        print(f"   Found {len(matches)} shape-valid constellation matches")
        
        if matches:
            print("   Matched constellations:")
            for i, match in enumerate(matches):
                hemisphere = match["hemisphere"]
                shape_score = match.get("shape_score", 0)
                print(f"     {i+1}. {match['constellation_name']}: {match['confidence']:.2f} confidence ({hemisphere}, shape: {shape_score:.2f})")
        else:
            print("   No shape-valid constellation patterns found")
            print("   This could indicate:")
            print("     - Image shows a different sky region")
            print("     - Shape constraints too restrictive")
            print("     - Need more flexible pattern matching")
        
        print(f"   Total processing time: {elapsed_time:.1f} seconds")
        print(f"   Check Output/shape_aware_constellations.jpg for results")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        raise

if __name__ == "__main__":
    main() 