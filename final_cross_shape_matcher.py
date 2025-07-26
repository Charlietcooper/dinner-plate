#!/usr/bin/env python3
"""
Final Cross Shape Constellation Matcher
Ensures proper cross shape for Southern Cross and allows more constellations
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

class FinalCrossShapeMatcher:
    """Final constellation matcher that ensures proper cross shapes and more constellations."""
    
    def __init__(self):
        self.constellation_patterns = self._create_cross_shape_patterns()
        self.detected_stars = None
        self.fov_info = None
        
    def _create_cross_shape_patterns(self) -> Dict:
        """Create constellation patterns with emphasis on proper cross shapes."""
        return {
            "Crux": {
                "name": "Southern Cross",
                "hemisphere": "southern",
                "pattern_type": "cross",
                "required_stars": [
                    {"name": "Acrux", "position": "center"},
                    {"name": "Mimosa", "position": "right"},
                    {"name": "Gacrux", "position": "top"},
                    {"name": "Delta", "position": "left"}
                ],
                "cross_validation": {
                    "horizontal_ratio": 1.0,  # Acrux to Mimosa
                    "vertical_ratio": 0.8,    # Acrux to Gacrux
                    "cross_angle_tolerance": 15  # degrees
                },
                "min_stars": 4,
                "min_confidence": 0.3,
                "description": "Southern Cross - proper cross shape"
            },
            "Carina": {
                "name": "Carina",
                "hemisphere": "southern",
                "pattern_type": "keel",
                "required_stars": [
                    {"name": "Canopus", "position": "base"},
                    {"name": "Avior", "position": "tip"},
                    {"name": "Aspidiske", "position": "side1"},
                    {"name": "Miaplacidus", "position": "side2"}
                ],
                "min_stars": 4,
                "min_confidence": 0.3,
                "description": "Carina - ship keel shape"
            },
            "Centaurus": {
                "name": "Centaurus",
                "hemisphere": "southern",
                "pattern_type": "centaur",
                "required_stars": [
                    {"name": "Alpha Cen", "position": "head"},
                    {"name": "Hadar", "position": "body"},
                    {"name": "Menkent", "position": "arm"},
                    {"name": "Muhlifain", "position": "leg"}
                ],
                "min_stars": 4,
                "min_confidence": 0.3,
                "description": "Centaurus - centaur shape"
            },
            "Musca": {
                "name": "Musca",
                "hemisphere": "southern",
                "pattern_type": "fly",
                "required_stars": [
                    {"name": "Alpha Mus", "position": "head"},
                    {"name": "Beta Mus", "position": "body"},
                    {"name": "Gamma Mus", "position": "wing1"},
                    {"name": "Delta Mus", "position": "wing2"}
                ],
                "min_stars": 4,
                "min_confidence": 0.25,
                "description": "Musca - fly shape"
            },
            "Puppis": {
                "name": "Puppis",
                "hemisphere": "southern",
                "pattern_type": "ship",
                "required_stars": [
                    {"name": "Naos", "position": "bow"},
                    {"name": "Tureis", "position": "stern"},
                    {"name": "Asmidiske", "position": "side1"},
                    {"name": "HD 64760", "position": "side2"}
                ],
                "min_stars": 4,
                "min_confidence": 0.25,
                "description": "Puppis - ship shape"
            },
            "Vela": {
                "name": "Vela",
                "hemisphere": "southern",
                "pattern_type": "sail",
                "required_stars": [
                    {"name": "Suhail", "position": "top"},
                    {"name": "Markeb", "position": "bottom"},
                    {"name": "Alsephina", "position": "left"},
                    {"name": "Tseen Ke", "position": "right"}
                ],
                "min_stars": 4,
                "min_confidence": 0.25,
                "description": "Vela - sail shape"
            },
            "Orion": {
                "name": "Orion",
                "hemisphere": "both",
                "pattern_type": "hunter",
                "required_stars": [
                    {"name": "Betelgeuse", "position": "shoulder"},
                    {"name": "Rigel", "position": "foot"},
                    {"name": "Bellatrix", "position": "shoulder"},
                    {"name": "Mintaka", "position": "belt"},
                    {"name": "Alnilam", "position": "belt"},
                    {"name": "Alnitak", "position": "belt"}
                ],
                "min_stars": 5,
                "min_confidence": 0.3,
                "description": "Orion - hunter with belt"
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
        """Find constellation patterns with proper cross shape validation."""
        logger.info("🔍 Searching for constellations with cross shape validation...")
        
        if not self.detected_stars:
            raise ValueError("No detected stars available")
        
        # Sort stars by brightness
        bright_stars = sorted(self.detected_stars, key=lambda s: s.get('brightness', 0), reverse=True)
        candidate_stars = bright_stars[:200]  # Use more stars for better coverage
        
        matches = []
        
        for const_name, const_data in self.constellation_patterns.items():
            logger.info(f"   Searching for {const_name}...")
            
            if const_data["pattern_type"] == "cross":
                # Special handling for cross-shaped constellations
                cross_matches = self._find_cross_pattern(candidate_stars, const_data)
                matches.extend(cross_matches)
            else:
                # General pattern matching for other constellations
                pattern_matches = self._find_general_pattern(candidate_stars, const_data)
                matches.extend(pattern_matches)
        
        # Apply lenient spatial constraints
        constrained_matches = self._apply_lenient_spatial_constraints(matches)
        
        # Sort by confidence
        constrained_matches.sort(key=lambda m: m["confidence"], reverse=True)
        
        logger.info(f"   Found {len(constrained_matches)} final constellation matches")
        return constrained_matches
    
    def _find_cross_pattern(self, stars: List[Dict], const_data: Dict) -> List[Dict]:
        """Find cross-shaped constellation patterns with proper validation."""
        matches = []
        required_stars = const_data["required_stars"]
        cross_validation = const_data["cross_validation"]
        
        # Try different center stars
        for center_star in stars[:50]:
            center_x, center_y = center_star["x"], center_star["y"]
            
            # Find stars in four directions to form a cross
            cross_stars = self._find_cross_stars(center_star, stars, cross_validation)
            
            if len(cross_stars) >= 4:
                # Validate cross shape
                cross_quality = self._validate_cross_shape(cross_stars, cross_validation)
                
                if cross_quality > 0.4:  # Minimum cross quality
                    confidence = self._calculate_cross_confidence(cross_stars, cross_quality)
                    
                    if confidence >= const_data["min_confidence"]:
                        matches.append({
                            "constellation": const_data["name"],
                            "constellation_name": const_data["name"],
                            "hemisphere": const_data["hemisphere"],
                            "description": const_data["description"],
                            "confidence": confidence,
                            "matched_stars": cross_stars,
                            "pattern_points": [(s["x"], s["y"]) for s in cross_stars],
                            "center_x": center_x,
                            "center_y": center_y,
                            "shape_score": cross_quality,
                            "pattern_type": "cross"
                        })
        
        return matches
    
    def _find_cross_stars(self, center_star: Dict, all_stars: List[Dict], cross_validation: Dict) -> List[Dict]:
        """Find stars in four directions to form a cross."""
        center_x, center_y = center_star["x"], center_star["y"]
        cross_stars = [center_star]
        
        # Find stars in four directions: right, top, left, bottom
        directions = [0, 90, 180, 270]  # degrees
        base_distance = 100  # pixels
        
        for direction in directions:
            best_star = None
            best_score = 0
            
            for star in all_stars:
                if star == center_star or star in cross_stars:
                    continue
                
                # Calculate distance and angle from center
                dx = star["x"] - center_x
                dy = star["y"] - center_y
                distance = math.sqrt(dx*dx + dy*dy)
                angle = math.degrees(math.atan2(dy, dx))
                
                # Normalize angle
                if angle < 0:
                    angle += 360
                
                # Check if star is in the right direction
                angle_diff = abs(angle - direction)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                
                if angle_diff < 30:  # Within 30 degrees of target direction
                    # Calculate score based on distance and angle
                    distance_score = 1.0 / (1.0 + abs(distance - base_distance) / base_distance)
                    angle_score = 1.0 / (1.0 + angle_diff / 30.0)
                    total_score = distance_score * angle_score
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_star = star
            
            if best_star:
                cross_stars.append(best_star)
        
        return cross_stars
    
    def _validate_cross_shape(self, cross_stars: List[Dict], cross_validation: Dict) -> float:
        """Validate that the stars form a proper cross shape."""
        if len(cross_stars) < 4:
            return 0.0
        
        center_star = cross_stars[0]
        center_x, center_y = center_star["x"], center_star["y"]
        
        # Calculate distances and angles
        distances = []
        angles = []
        
        for star in cross_stars[1:]:
            dx = star["x"] - center_x
            dy = star["y"] - center_y
            distance = math.sqrt(dx*dx + dy*dy)
            angle = math.degrees(math.atan2(dy, dx))
            
            if angle < 0:
                angle += 360
            
            distances.append(distance)
            angles.append(angle)
        
        # Check for perpendicular lines (cross shape)
        perpendicular_pairs = 0
        total_pairs = 0
        
        for i in range(len(angles)):
            for j in range(i+1, len(angles)):
                angle_diff = abs(angles[i] - angles[j])
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                
                # Check if angles are approximately perpendicular (90° ± tolerance)
                if abs(angle_diff - 90) < cross_validation["cross_angle_tolerance"]:
                    perpendicular_pairs += 1
                total_pairs += 1
        
        if total_pairs == 0:
            return 0.0
        
        cross_quality = perpendicular_pairs / total_pairs
        
        # Also check distance ratios
        if len(distances) >= 2:
            distance_ratios = []
            for i in range(len(distances)):
                for j in range(i+1, len(distances)):
                    if distances[j] > 0:
                        ratio = distances[i] / distances[j]
                        distance_ratios.append(ratio)
            
            # Check if distance ratios are reasonable
            avg_ratio = np.mean(distance_ratios)
            ratio_quality = 1.0 / (1.0 + abs(avg_ratio - 1.0))
            
            # Combine cross quality and ratio quality
            final_quality = 0.7 * cross_quality + 0.3 * ratio_quality
            return final_quality
        
        return cross_quality
    
    def _calculate_cross_confidence(self, cross_stars: List[Dict], cross_quality: float) -> float:
        """Calculate confidence for cross-shaped constellation."""
        # Base confidence from cross quality
        confidence = cross_quality
        
        # Add brightness bonus
        brightness_bonus = sum(s.get("brightness", 0) for s in cross_stars) / (len(cross_stars) * 255)
        confidence = 0.6 * confidence + 0.4 * brightness_bonus
        
        return confidence
    
    def _find_general_pattern(self, stars: List[Dict], const_data: Dict) -> List[Dict]:
        """Find general constellation patterns."""
        matches = []
        required_stars = const_data["required_stars"]
        min_stars = const_data["min_stars"]
        
        # Try different starting stars
        for start_star in stars[:30]:
            pattern_matches = self._try_general_pattern_from_star(start_star, stars, required_stars, min_stars)
            matches.extend(pattern_matches)
        
        return matches
    
    def _try_general_pattern_from_star(self, start_star: Dict, all_stars: List[Dict], 
                                      required_stars: List[Dict], min_stars: int) -> List[Dict]:
        """Try to match a general constellation pattern."""
        matches = []
        
        start_x, start_y = start_star["x"], start_star["y"]
        
        # Try different second stars
        for second_star in all_stars:
            if second_star == start_star:
                continue
            
            # Calculate base distance
            dx = second_star["x"] - start_x
            dy = second_star["y"] - start_y
            base_distance = math.sqrt(dx*dx + dy*dy)
            
            if base_distance < 50 or base_distance > 400:
                continue
            
            # Build pattern
            pattern_stars = [start_star, second_star]
            pattern_points = [(start_x, start_y), (second_star["x"], second_star["y"])]
            
            # Try to add more stars
            for i in range(2, min(len(required_stars), 6)):
                best_star = None
                best_score = 0
                
                for candidate_star in all_stars:
                    if candidate_star in pattern_stars:
                        continue
                    
                    # Calculate distance from previous star
                    prev_star = pattern_stars[-1]
                    dx = candidate_star["x"] - prev_star["x"]
                    dy = candidate_star["y"] - prev_star["y"]
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    # Score based on reasonable distance
                    if 30 < distance < 300:
                        score = 1.0 / (1.0 + abs(distance - base_distance) / base_distance)
                        if score > best_score:
                            best_score = score
                            best_star = candidate_star
                
                if best_star:
                    pattern_stars.append(best_star)
                    pattern_points.append((best_star["x"], best_star["y"]))
                else:
                    break
            
            if len(pattern_stars) >= min_stars:
                # Calculate confidence
                brightness_bonus = sum(s.get("brightness", 0) for s in pattern_stars) / (len(pattern_stars) * 255)
                pattern_quality = len(pattern_stars) / len(required_stars)
                confidence = 0.5 * pattern_quality + 0.5 * brightness_bonus
                
                if confidence > 0.2:
                    matches.append({
                        "constellation": "General",
                        "constellation_name": "General Pattern",
                        "hemisphere": "southern",  # Default for this image
                        "description": "General constellation pattern",
                        "confidence": confidence,
                        "matched_stars": pattern_stars,
                        "pattern_points": pattern_points,
                        "center_x": np.mean([s["x"] for s in pattern_stars]),
                        "center_y": np.mean([s["y"] for s in pattern_stars]),
                        "shape_score": pattern_quality,
                        "pattern_type": "general"
                    })
        
        return matches
    
    def _apply_lenient_spatial_constraints(self, matches: List[Dict]) -> List[Dict]:
        """Apply lenient spatial constraints to allow more constellations."""
        logger.info("🌍 Applying lenient spatial constraints...")
        
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
        
        # Rule 3: Very lenient shape quality validation
        valid_matches = []
        for match in matches:
            shape_score = match.get("shape_score", 0)
            if shape_score > 0.1:  # Very low threshold
                valid_matches.append(match)
                logger.info(f"   ✅ Added {match['constellation_name']} (shape score: {shape_score:.2f})")
            else:
                logger.info(f"   ❌ Rejected {match['constellation_name']} (poor shape match: {shape_score:.2f})")
        
        # Rule 4: Very lenient spatial separation
        final_matches = []
        for match in valid_matches:
            is_valid = True
            for final_match in final_matches:
                distance = self._calculate_constellation_distance(match, final_match)
                if distance < 100:  # Very small minimum separation
                    is_valid = False
                    logger.info(f"   ❌ Rejected {match['constellation_name']} (too close to {final_match['constellation_name']})")
                    break
            
            if is_valid:
                final_matches.append(match)
                logger.info(f"   ✅ Added {match['constellation_name']} (spatially separated)")
        
        # Rule 5: High FOV limits
        max_constellations = self._get_high_max_constellations_for_fov()
        if len(final_matches) > max_constellations:
            logger.info(f"   Limiting to {max_constellations} constellations based on FOV")
            final_matches = final_matches[:max_constellations]
        
        return final_matches
    
    def _calculate_constellation_distance(self, match1: Dict, match2: Dict) -> float:
        """Calculate distance between two constellation centers."""
        dx = match1["center_x"] - match2["center_x"]
        dy = match1["center_y"] - match2["center_y"]
        return math.sqrt(dx*dx + dy*dy)
    
    def _get_high_max_constellations_for_fov(self) -> int:
        """Get high maximum number of constellations based on FOV."""
        fov_estimate = self.fov_info.get("fov_estimate", "unknown")
        
        if "narrow_field" in fov_estimate:
            return 6  # High limit for narrow field
        elif "medium_field" in fov_estimate:
            return 8  # High limit for medium field
        else:
            return 10  # High limit for wide field
    
    def create_annotated_image(self, matches: List[Dict], output_path: str = "Output/final_cross_shape_constellations.jpg"):
        """Create annotated image with final cross shape patterns."""
        logger.info("🎨 Creating final cross shape annotated image...")
        
        # Load original image
        original_image = cv2.imread("Input/test-1.jpg")
        if original_image is None:
            logger.error("❌ Could not load original image")
            return
        
        # Create annotation overlay
        annotated_image = original_image.copy()
        
        # Draw matched constellations
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]
        
        for i, match in enumerate(matches):
            color = colors[i % len(colors)]
            self._draw_final_constellation_pattern(annotated_image, match, color)
        
        # Add information overlay
        annotated_image = self._add_final_info_overlay(annotated_image, matches)
        
        # Save result
        cv2.imwrite(output_path, annotated_image)
        logger.info(f"   ✅ Saved annotated image: {output_path}")
    
    def _draw_final_constellation_pattern(self, image: np.ndarray, match: Dict, color: Tuple[int, int, int]):
        """Draw a final constellation pattern on the image."""
        pattern_points = match["pattern_points"]
        pattern_type = match.get("pattern_type", "general")
        
        if pattern_type == "cross":
            # Draw cross pattern
            center_point = pattern_points[0]
            for i, point in enumerate(pattern_points[1:], 1):
                pt1 = tuple(map(int, center_point))
                pt2 = tuple(map(int, point))
                cv2.line(image, pt1, pt2, color, 4)
        else:
            # Draw general pattern
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
            pattern_type = match.get("pattern_type", "general")
            label = f"{match['constellation_name']} ({match['confidence']:.2f}) [{hemisphere}]"
            cv2.putText(image, label, 
                       (text_pos[0] + 15, text_pos[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            
            # Add shape score and pattern type
            shape_label = f"Shape: {shape_score:.2f} ({pattern_type})"
            cv2.putText(image, shape_label,
                       (text_pos[0] + 15, text_pos[1] + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    def _add_final_info_overlay(self, image: np.ndarray, matches: List[Dict]) -> np.ndarray:
        """Add final information overlay to the image."""
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
        title = "Final Cross Shape Constellation Matching"
        draw.text((20, 20), title, fill=(255, 255, 255), font=font)
        
        # Add processing information
        y_offset = 80
        info_lines = [
            f"Stars analyzed: {len(self.detected_stars)}",
            f"Final matches: {len(matches)}",
            f"FOV estimate: {self.fov_info.get('fov_estimate', 'unknown')}",
            f"Method: Cross shape validation with lenient constraints"
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
            pattern_type = match.get("pattern_type", "general")
            num_stars = len(match["matched_stars"])
            
            line = f"• {const_name}: {confidence:.2f} confidence ({hemisphere}, {num_stars} stars)"
            draw.text((20, y_offset), line, fill=(0, 255, 255), font=small_font)
            y_offset += 25
            line2 = f"  Shape: {shape_score:.2f} ({pattern_type})"
            draw.text((20, y_offset), line2, fill=(255, 165, 0), font=small_font)
            y_offset += 30
        
        # Add constraint information
        y_offset += 20
        draw.text((20, y_offset), "Applied improvements:", 
                 fill=(255, 165, 0), font=small_font)
        y_offset += 25
        draw.text((20, y_offset), "• Proper cross shape validation", 
                 fill=(255, 165, 0), font=small_font)
        y_offset += 25
        draw.text((20, y_offset), "• Lenient spatial constraints", 
                 fill=(255, 165, 0), font=small_font)
        y_offset += 25
        draw.text((20, y_offset), "• More constellations for orientation", 
                 fill=(255, 165, 0), font=small_font)
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def main():
    """Test final cross shape constellation matching."""
    print("🔍 Final Cross Shape Constellation Matcher")
    print("=" * 60)
    print("Ensures proper cross shapes and allows more constellations")
    print("=" * 60)
    
    start_time = time.time()
    
    # Create matcher
    matcher = FinalCrossShapeMatcher()
    
    try:
        # Load processed data
        matcher.load_processed_data()
        
        # Find constellation patterns
        matches = matcher.find_constellation_patterns()
        
        # Create annotated image
        matcher.create_annotated_image(matches)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ Final cross shape matching complete!")
        print(f"   Found {len(matches)} final constellation matches")
        
        if matches:
            print("   Matched constellations:")
            for i, match in enumerate(matches):
                hemisphere = match["hemisphere"]
                shape_score = match.get("shape_score", 0)
                pattern_type = match.get("pattern_type", "general")
                print(f"     {i+1}. {match['constellation_name']}: {match['confidence']:.2f} confidence ({hemisphere}, shape: {shape_score:.2f}, type: {pattern_type})")
        else:
            print("   No final constellation patterns found")
            print("   This could indicate:")
            print("     - Image shows a different sky region")
            print("     - Need to adjust cross validation parameters")
            print("     - Need more constellation patterns")
        
        print(f"   Total processing time: {elapsed_time:.1f} seconds")
        print(f"   Check Output/final_cross_shape_constellations.jpg for results")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        raise

if __name__ == "__main__":
    main() 