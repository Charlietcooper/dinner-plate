#!/usr/bin/env python3
"""
Final Spatial Constrained Constellation Matcher
Complete solution with duplicate prevention and proper adjacency validation
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

class FinalSpatialConstellationMatcher:
    """Final constellation matcher with complete spatial constraints."""
    
    def __init__(self):
        self.constellation_patterns = self._create_constellation_patterns()
        self.spatial_constraints = self._create_spatial_constraints()
        self.adjacency_rules = self._create_adjacency_rules()
        self.detected_stars = None
        self.fov_info = None
        
    def _create_constellation_patterns(self) -> Dict:
        """Create constellation patterns with hemisphere information."""
        return {
            "Crux": {
                "name": "Southern Cross",
                "hemisphere": "southern",
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
            "Carina": {
                "name": "Carina",
                "hemisphere": "southern",
                "pattern": [
                    {"distance_ratio": 1.0, "angle": 0},      # Canopus to Avior
                    {"distance_ratio": 0.9, "angle": 45},     # Avior to Aspidiske
                    {"distance_ratio": 0.8, "angle": 90}      # Aspidiske to Miaplacidus
                ],
                "min_stars": 4,
                "min_confidence": 0.5,
                "description": "Carina - keel of the ship"
            },
            "Centaurus": {
                "name": "Centaurus",
                "hemisphere": "southern",
                "pattern": [
                    {"distance_ratio": 1.0, "angle": 0},      # Alpha to Beta Centauri
                    {"distance_ratio": 0.8, "angle": 60},     # Beta to Gamma Centauri
                    {"distance_ratio": 0.7, "angle": 120}     # Gamma to Delta Centauri
                ],
                "min_stars": 4,
                "min_confidence": 0.5,
                "description": "Centaurus - the centaur"
            },
            "Orion": {
                "name": "Orion",
                "hemisphere": "both",  # Visible from both hemispheres
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
                "hemisphere": "northern",
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
                "hemisphere": "northern",
                "pattern": [
                    {"distance_ratio": 1.0, "angle": 0},      # W point 1 to 2
                    {"distance_ratio": 0.8, "angle": 45},     # W point 2 to 3
                    {"distance_ratio": 0.8, "angle": 315},    # W point 3 to 4
                    {"distance_ratio": 1.0, "angle": 0}       # W point 4 to 5
                ],
                "min_stars": 5,
                "min_confidence": 0.5,
                "description": "Cassiopeia - W or M shape"
            },
            "Leo": {
                "name": "Leo",
                "hemisphere": "both",
                "pattern": [
                    {"distance_ratio": 1.0, "angle": 0},      # Regulus to Denebola
                    {"distance_ratio": 0.7, "angle": 45},     # Denebola to Algieba
                    {"distance_ratio": 0.7, "angle": 90},     # Algieba to Zosma
                    {"distance_ratio": 0.7, "angle": 135}     # Zosma to Chertan
                ],
                "min_stars": 4,
                "min_confidence": 0.5,
                "description": "Leo - the lion"
            },
            "Virgo": {
                "name": "Virgo",
                "hemisphere": "both",
                "pattern": [
                    {"distance_ratio": 1.0, "angle": 0},      # Spica to Vindemiatrix
                    {"distance_ratio": 0.8, "angle": 45},     # Vindemiatrix to Porrima
                    {"distance_ratio": 0.7, "angle": 90}      # Porrima to Auva
                ],
                "min_stars": 4,
                "min_confidence": 0.5,
                "description": "Virgo - the maiden"
            }
        }
    
    def _create_spatial_constraints(self) -> Dict:
        """Create spatial constraints based on constellation positions."""
        return {
            "hemisphere_rules": {
                "northern_only": ["Ursa Major", "Cassiopeia", "Ursa Minor", "Draco", "Cepheus"],
                "southern_only": ["Crux", "Carina", "Centaurus", "Musca", "Pavo", "Tucana"],
                "both_hemispheres": ["Orion", "Leo", "Virgo", "Gemini", "Taurus", "Canis Major"]
            },
            "seasonal_constraints": {
                "spring": ["Leo", "Virgo", "Ursa Major"],
                "summer": ["Orion", "Canis Major", "Gemini"],
                "autumn": ["Pegasus", "Andromeda", "Cassiopeia"],
                "winter": ["Orion", "Taurus", "Ursa Major"]
            }
        }
    
    def _create_adjacency_rules(self) -> Dict:
        """Create adjacency rules based on actual constellation positions from the chart."""
        return {
            "Crux": ["Carina", "Centaurus", "Musca"],  # Southern Cross adjacent to Carina, Centaurus
            "Carina": ["Crux", "Centaurus", "Puppis", "Vela"],  # Carina adjacent to Crux, Centaurus
            "Centaurus": ["Crux", "Carina", "Lupus", "Circinus"],  # Centaurus adjacent to Crux, Carina
            "Orion": ["Taurus", "Gemini", "Canis Major", "Canis Minor", "Eridanus"],  # Orion adjacent to Taurus, Gemini
            "Ursa Major": ["Cassiopeia", "Draco", "Leo Minor", "Canes Venatici", "Coma Berenices"],  # Big Dipper adjacent to Cassiopeia
            "Cassiopeia": ["Ursa Major", "Perseus", "Andromeda", "Cepheus"],  # Cassiopeia adjacent to Ursa Major
            "Leo": ["Virgo", "Cancer", "Ursa Major", "Coma Berenices", "Sextans"],  # Leo adjacent to Virgo
            "Virgo": ["Leo", "Libra", "Coma Berenices", "Booetes", "Corvus"]  # Virgo adjacent to Leo
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
        """Find constellation patterns with complete spatial constraints."""
        logger.info("🔍 Searching for constellation patterns with complete spatial constraints...")
        
        if not self.detected_stars:
            raise ValueError("No detected stars available")
        
        # First pass: find all potential matches
        all_matches = []
        bright_stars = sorted(self.detected_stars, key=lambda s: s.get('brightness', 0), reverse=True)
        candidate_stars = bright_stars[:50]
        
        for const_name, const_data in self.constellation_patterns.items():
            logger.info(f"   Searching for {const_name}...")
            
            pattern_matches = self._find_pattern_matches(candidate_stars, const_data)
            
            # Filter by minimum confidence
            filtered_matches = [m for m in pattern_matches if m["confidence"] >= const_data["min_confidence"]]
            
            for match in filtered_matches:
                all_matches.append({
                    "constellation": const_name,
                    "constellation_name": const_data["name"],
                    "hemisphere": const_data["hemisphere"],
                    "description": const_data["description"],
                    "confidence": match["confidence"],
                    "matched_stars": match["stars"],
                    "pattern_points": match["pattern_points"],
                    "center_x": np.mean([s["x"] for s in match["stars"]]),
                    "center_y": np.mean([s["y"] for s in match["stars"]])
                })
        
        # Apply complete spatial constraints
        constrained_matches = self._apply_complete_spatial_constraints(all_matches)
        
        # Sort by confidence
        constrained_matches.sort(key=lambda m: m["confidence"], reverse=True)
        
        logger.info(f"   Found {len(constrained_matches)} final constellation matches")
        return constrained_matches
    
    def _apply_complete_spatial_constraints(self, matches: List[Dict]) -> List[Dict]:
        """Apply complete spatial constraints including duplicate prevention."""
        logger.info("🌍 Applying complete spatial constraints...")
        
        if not matches:
            return []
        
        # Group by hemisphere
        northern_matches = [m for m in matches if m["hemisphere"] == "northern"]
        southern_matches = [m for m in matches if m["hemisphere"] == "southern"]
        both_matches = [m for m in matches if m["hemisphere"] == "both"]
        
        logger.info(f"   Northern: {len(northern_matches)}, Southern: {len(southern_matches)}, Both: {len(both_matches)}")
        
        # Rule 1: Cannot have northern and southern constellations together
        if northern_matches and southern_matches:
            logger.info("   ❌ Detected both northern and southern constellations - applying hemisphere constraint")
            
            # Choose the hemisphere with more high-confidence matches
            northern_score = sum(m["confidence"] for m in northern_matches)
            southern_score = sum(m["confidence"] for m in southern_matches)
            
            if northern_score > southern_score:
                logger.info(f"   ✅ Keeping northern hemisphere (score: {northern_score:.2f})")
                matches = northern_matches + both_matches
            else:
                logger.info(f"   ✅ Keeping southern hemisphere (score: {southern_score:.2f})")
                matches = southern_matches + both_matches
        
        # Rule 2: Remove duplicate constellations (keep highest confidence)
        matches = self._remove_duplicate_constellations(matches)
        
        # Rule 3: Check adjacency constraints
        valid_combinations = self._check_adjacency_constraints(matches)
        
        # Rule 4: Limit total constellations based on FOV
        max_constellations = self._get_max_constellations_for_fov()
        if len(valid_combinations) > max_constellations:
            logger.info(f"   Limiting to {max_constellations} constellations based on FOV")
            valid_combinations = valid_combinations[:max_constellations]
        
        return valid_combinations
    
    def _remove_duplicate_constellations(self, matches: List[Dict]) -> List[Dict]:
        """Remove duplicate constellations, keeping the highest confidence match."""
        logger.info("🔄 Removing duplicate constellations...")
        
        constellation_groups = {}
        
        # Group by constellation name
        for match in matches:
            const_name = match["constellation"]
            if const_name not in constellation_groups:
                constellation_groups[const_name] = []
            constellation_groups[const_name].append(match)
        
        # Keep only the highest confidence match for each constellation
        unique_matches = []
        for const_name, group_matches in constellation_groups.items():
            best_match = max(group_matches, key=lambda m: m["confidence"])
            unique_matches.append(best_match)
            logger.info(f"   ✅ Kept {const_name} (confidence: {best_match['confidence']:.2f})")
        
        logger.info(f"   Reduced from {len(matches)} to {len(unique_matches)} unique constellations")
        return unique_matches
    
    def _check_adjacency_constraints(self, matches: List[Dict]) -> List[Dict]:
        """Check that constellations are actually adjacent to each other."""
        if len(matches) <= 1:
            return matches
        
        # Sort by confidence
        matches.sort(key=lambda m: m["confidence"], reverse=True)
        
        valid_matches = [matches[0]]  # Always keep the highest confidence match
        logger.info(f"   ✅ Added {matches[0]['constellation_name']} (highest confidence)")
        
        for match in matches[1:]:
            const_name = match["constellation"]
            is_valid = True
            rejection_reason = ""
            
            # Check if this constellation is adjacent to any already accepted constellation
            for valid_match in valid_matches:
                valid_const = valid_match["constellation"]
                
                # Check adjacency rules
                if const_name in self.adjacency_rules.get(valid_const, []):
                    # They are adjacent - this is good
                    continue
                elif valid_const in self.adjacency_rules.get(const_name, []):
                    # They are adjacent - this is good
                    continue
                else:
                    # Check if they're far enough apart in the image
                    distance = self._calculate_constellation_distance(match, valid_match)
                    if distance < 250:  # Increased minimum distance
                        rejection_reason = f"too close to {valid_const} (distance: {distance:.1f})"
                        is_valid = False
                        break
            
            if is_valid:
                valid_matches.append(match)
                logger.info(f"   ✅ Added {const_name} (adjacent or well-separated)")
            else:
                logger.info(f"   ❌ Rejected {const_name} ({rejection_reason})")
        
        return valid_matches
    
    def _calculate_constellation_distance(self, match1: Dict, match2: Dict) -> float:
        """Calculate distance between two constellation centers."""
        dx = match1["center_x"] - match2["center_x"]
        dy = match1["center_y"] - match2["center_y"]
        return math.sqrt(dx*dx + dy*dy)
    
    def _get_max_constellations_for_fov(self) -> int:
        """Get maximum number of constellations based on FOV."""
        fov_estimate = self.fov_info.get("fov_estimate", "unknown")
        
        if "narrow_field" in fov_estimate:
            return 2  # Very narrow field - only 1-2 constellations
        elif "medium_field" in fov_estimate:
            return 3  # Medium field - 2-3 constellations
        else:
            return 4  # Wide field - up to 4 constellations
    
    def _find_pattern_matches(self, stars: List[Dict], const_data: Dict) -> List[Dict]:
        """Find matches for a specific constellation pattern."""
        matches = []
        pattern = const_data["pattern"]
        min_stars = const_data["min_stars"]
        
        # Try different starting stars
        for start_star in stars[:10]:
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
            if base_distance < 100:
                continue
            if base_distance > 300:
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
                    if distance_error < 0.3 and angle_error < 0.2:
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
                if confidence > 0.4:
                    matches.append({
                        "confidence": confidence,
                        "stars": pattern_stars,
                        "pattern_points": pattern_points
                    })
        
        return matches
    
    def create_annotated_image(self, matches: List[Dict], output_path: str = "Output/final_spatial_constellations.jpg"):
        """Create annotated image with final spatially-constrained constellation patterns."""
        logger.info("🎨 Creating final spatially-constrained annotated image...")
        
        # Load original image
        original_image = cv2.imread("Input/test-1.jpg")
        if original_image is None:
            logger.error("❌ Could not load original image")
            return
        
        # Create annotation overlay
        annotated_image = original_image.copy()
        
        # Draw matched constellations
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
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
        
        # Add constellation name, confidence, and hemisphere
        if pattern_points:
            text_pos = tuple(map(int, pattern_points[0]))
            hemisphere = match["hemisphere"]
            label = f"{match['constellation_name']} ({match['confidence']:.2f}) [{hemisphere}]"
            cv2.putText(image, label, 
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
        title = "Final Spatially-Constrained Constellation Matching"
        draw.text((20, 20), title, fill=(255, 255, 255), font=font)
        
        # Add processing information
        y_offset = 80
        info_lines = [
            f"Stars analyzed: {len(self.detected_stars)}",
            f"Final valid matches: {len(matches)}",
            f"FOV estimate: {self.fov_info.get('fov_estimate', 'unknown')}",
            f"Method: Pattern matching with complete spatial constraints"
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
            num_stars = len(match["matched_stars"])
            
            line = f"• {const_name}: {confidence:.2f} confidence ({hemisphere}, {num_stars} stars)"
            draw.text((20, y_offset), line, fill=(0, 255, 255), font=small_font)
            y_offset += 30
        
        # Add constraint information
        y_offset += 20
        draw.text((20, y_offset), "Applied constraints:", 
                 fill=(255, 165, 0), font=small_font)
        y_offset += 25
        draw.text((20, y_offset), "• Hemisphere consistency", 
                 fill=(255, 165, 0), font=small_font)
        y_offset += 25
        draw.text((20, y_offset), "• Duplicate prevention", 
                 fill=(255, 165, 0), font=small_font)
        y_offset += 25
        draw.text((20, y_offset), "• Adjacency validation", 
                 fill=(255, 165, 0), font=small_font)
        y_offset += 25
        draw.text((20, y_offset), "• FOV-based limits", 
                 fill=(255, 165, 0), font=small_font)
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def main():
    """Test final spatially-constrained constellation matching."""
    print("🌍 Final Spatially-Constrained Constellation Matcher")
    print("=" * 60)
    print("Complete solution with duplicate prevention and proper adjacency validation")
    print("=" * 60)
    
    start_time = time.time()
    
    # Create matcher
    matcher = FinalSpatialConstellationMatcher()
    
    try:
        # Load processed data
        matcher.load_processed_data()
        
        # Find constellation patterns
        matches = matcher.find_constellation_patterns()
        
        # Create annotated image
        matcher.create_annotated_image(matches)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ Final spatially-constrained matching complete!")
        print(f"   Found {len(matches)} final constellation matches")
        
        if matches:
            print("   Matched constellations:")
            for i, match in enumerate(matches):
                hemisphere = match["hemisphere"]
                print(f"     {i+1}. {match['constellation_name']}: {match['confidence']:.2f} confidence ({hemisphere})")
        else:
            print("   No final constellation patterns found")
            print("   This could indicate:")
            print("     - Image shows a different sky region")
            print("     - Spatial constraints too restrictive")
            print("     - Need more sophisticated pattern recognition")
        
        print(f"   Total processing time: {elapsed_time:.1f} seconds")
        print(f"   Check Output/final_spatial_constellations.jpg for results")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        raise

if __name__ == "__main__":
    main() 