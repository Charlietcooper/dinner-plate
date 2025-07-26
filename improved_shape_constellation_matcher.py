#!/usr/bin/env python3
"""
Improved Shape Constellation Matcher
Fixes shape matching and allows more constellations for better spatial orientation
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

class ImprovedShapeConstellationMatcher:
    """Improved constellation matcher with better shape matching and more constellations."""
    
    def __init__(self):
        self.constellation_shapes = self._create_improved_constellation_shapes()
        self.detected_stars = None
        self.fov_info = None
        
    def _create_improved_constellation_shapes(self) -> Dict:
        """Create improved constellation shapes with proper patterns."""
        return {
            "Crux": {
                "name": "Southern Cross",
                "hemisphere": "southern",
                "shape_type": "cross",
                "pattern": [
                    # Acrux (alpha) to Mimosa (beta) - horizontal arm
                    {"from": 0, "to": 1, "distance_ratio": 1.0, "angle": 0},
                    # Mimosa (beta) to Gacrux (gamma) - vertical arm
                    {"from": 1, "to": 2, "distance_ratio": 0.8, "angle": 90},
                    # Gacrux (gamma) to Delta Crucis - horizontal arm
                    {"from": 2, "to": 3, "distance_ratio": 0.7, "angle": 180},
                    # Delta Crucis back to Acrux - vertical arm
                    {"from": 3, "to": 0, "distance_ratio": 0.9, "angle": 270}
                ],
                "star_positions": [
                    {"name": "Acrux", "x": 0, "y": 0},      # Center
                    {"name": "Mimosa", "x": 1, "y": 0},     # Right
                    {"name": "Gacrux", "x": 1, "y": 0.8},   # Top
                    {"name": "Delta", "x": 0, "y": 0.8}     # Left
                ],
                "min_stars": 4,
                "min_confidence": 0.5,
                "description": "Southern Cross - distinctive cross shape"
            },
            "Carina": {
                "name": "Carina",
                "hemisphere": "southern",
                "shape_type": "ship_keel",
                "pattern": [
                    {"from": 0, "to": 1, "distance_ratio": 1.0, "angle": 0},
                    {"from": 1, "to": 2, "distance_ratio": 0.9, "angle": 45},
                    {"from": 2, "to": 3, "distance_ratio": 0.8, "angle": 90}
                ],
                "star_positions": [
                    {"name": "Canopus", "x": 0, "y": 0},
                    {"name": "Avior", "x": 1, "y": 0},
                    {"name": "Aspidiske", "x": 1.4, "y": 0.7},
                    {"name": "Miaplacidus", "x": 1.4, "y": 1.4}
                ],
                "min_stars": 4,
                "min_confidence": 0.4,
                "description": "Carina - keel of the ship"
            },
            "Centaurus": {
                "name": "Centaurus",
                "hemisphere": "southern",
                "shape_type": "centaur",
                "pattern": [
                    {"from": 0, "to": 1, "distance_ratio": 1.0, "angle": 0},
                    {"from": 1, "to": 2, "distance_ratio": 0.8, "angle": 60},
                    {"from": 2, "to": 3, "distance_ratio": 0.7, "angle": 120}
                ],
                "star_positions": [
                    {"name": "Alpha Cen", "x": 0, "y": 0},
                    {"name": "Hadar", "x": 1, "y": 0},
                    {"name": "Menkent", "x": 1.4, "y": 0.7},
                    {"name": "Muhlifain", "x": 1.8, "y": 1.2}
                ],
                "min_stars": 4,
                "min_confidence": 0.4,
                "description": "Centaurus - the centaur"
            },
            "Orion": {
                "name": "Orion",
                "hemisphere": "both",
                "shape_type": "hunter",
                "pattern": [
                    # Belt stars
                    {"from": 0, "to": 1, "distance_ratio": 1.0, "angle": 0},
                    {"from": 1, "to": 2, "distance_ratio": 1.0, "angle": 0},
                    # Shoulders
                    {"from": 0, "to": 3, "distance_ratio": 1.2, "angle": 90},
                    {"from": 2, "to": 4, "distance_ratio": 1.2, "angle": 90},
                    # Feet
                    {"from": 1, "to": 5, "distance_ratio": 1.5, "angle": 270}
                ],
                "star_positions": [
                    {"name": "Mintaka", "x": 0, "y": 0},    # Belt left
                    {"name": "Alnilam", "x": 1, "y": 0},    # Belt center
                    {"name": "Alnitak", "x": 2, "y": 0},    # Belt right
                    {"name": "Betelgeuse", "x": 0, "y": 1.2}, # Shoulder left
                    {"name": "Bellatrix", "x": 2, "y": 1.2},  # Shoulder right
                    {"name": "Rigel", "x": 1, "y": -1.5}    # Foot
                ],
                "min_stars": 5,
                "min_confidence": 0.4,
                "description": "Orion - hunter with distinctive belt"
            },
            "Ursa Major": {
                "name": "Big Dipper",
                "hemisphere": "northern",
                "shape_type": "dipper",
                "pattern": [
                    # Handle
                    {"from": 0, "to": 1, "distance_ratio": 1.0, "angle": 0},
                    {"from": 1, "to": 2, "distance_ratio": 1.0, "angle": 0},
                    # Bowl
                    {"from": 2, "to": 3, "distance_ratio": 0.8, "angle": 90},
                    {"from": 3, "to": 4, "distance_ratio": 0.8, "angle": 0},
                    {"from": 4, "to": 5, "distance_ratio": 0.8, "angle": 270}
                ],
                "star_positions": [
                    {"name": "Alkaid", "x": 0, "y": 0},     # Handle end
                    {"name": "Mizar", "x": 1, "y": 0},      # Handle
                    {"name": "Alioth", "x": 2, "y": 0},     # Handle to bowl
                    {"name": "Megrez", "x": 2, "y": 0.8},   # Bowl left
                    {"name": "Phecda", "x": 2.8, "y": 0.8}, # Bowl right
                    {"name": "Merak", "x": 2.8, "y": 0}     # Bowl to handle
                ],
                "min_stars": 5,
                "min_confidence": 0.4,
                "description": "Big Dipper - distinctive dipper shape"
            },
            "Cassiopeia": {
                "name": "Cassiopeia",
                "hemisphere": "northern",
                "shape_type": "w_shape",
                "pattern": [
                    {"from": 0, "to": 1, "distance_ratio": 1.0, "angle": 0},
                    {"from": 1, "to": 2, "distance_ratio": 0.8, "angle": 45},
                    {"from": 2, "to": 3, "distance_ratio": 0.8, "angle": 315},
                    {"from": 3, "to": 4, "distance_ratio": 1.0, "angle": 0}
                ],
                "star_positions": [
                    {"name": "Schedar", "x": 0, "y": 0},
                    {"name": "Caph", "x": 1, "y": 0},
                    {"name": "Cih", "x": 1.4, "y": 0.7},
                    {"name": "Ruchbah", "x": 2.4, "y": 0.7},
                    {"name": "Segin", "x": 3.4, "y": 0}
                ],
                "min_stars": 5,
                "min_confidence": 0.4,
                "description": "Cassiopeia - W or M shape"
            },
            "Leo": {
                "name": "Leo",
                "hemisphere": "both",
                "shape_type": "sickle",
                "pattern": [
                    {"from": 0, "to": 1, "distance_ratio": 1.0, "angle": 0},
                    {"from": 1, "to": 2, "distance_ratio": 0.7, "angle": 45},
                    {"from": 2, "to": 3, "distance_ratio": 0.7, "angle": 90},
                    {"from": 3, "to": 4, "distance_ratio": 0.7, "angle": 135}
                ],
                "star_positions": [
                    {"name": "Regulus", "x": 0, "y": 0},
                    {"name": "Denebola", "x": 1, "y": 0},
                    {"name": "Algieba", "x": 1.35, "y": 0.7},
                    {"name": "Zosma", "x": 1.7, "y": 1.4},
                    {"name": "Chertan", "x": 2.05, "y": 2.1}
                ],
                "min_stars": 4,
                "min_confidence": 0.4,
                "description": "Leo - sickle shape"
            },
            "Virgo": {
                "name": "Virgo",
                "hemisphere": "both",
                "shape_type": "maiden",
                "pattern": [
                    {"from": 0, "to": 1, "distance_ratio": 1.0, "angle": 0},
                    {"from": 1, "to": 2, "distance_ratio": 0.8, "angle": 45},
                    {"from": 2, "to": 3, "distance_ratio": 0.7, "angle": 90}
                ],
                "star_positions": [
                    {"name": "Spica", "x": 0, "y": 0},
                    {"name": "Vindemiatrix", "x": 1, "y": 0},
                    {"name": "Porrima", "x": 1.4, "y": 0.7},
                    {"name": "Auva", "x": 1.7, "y": 1.4}
                ],
                "min_stars": 4,
                "min_confidence": 0.4,
                "description": "Virgo - the maiden"
            },
            "Musca": {
                "name": "Musca",
                "hemisphere": "southern",
                "shape_type": "fly",
                "pattern": [
                    {"from": 0, "to": 1, "distance_ratio": 1.0, "angle": 0},
                    {"from": 1, "to": 2, "distance_ratio": 0.8, "angle": 60},
                    {"from": 2, "to": 3, "distance_ratio": 0.7, "angle": 120}
                ],
                "star_positions": [
                    {"name": "Alpha Mus", "x": 0, "y": 0},
                    {"name": "Beta Mus", "x": 1, "y": 0},
                    {"name": "Gamma Mus", "x": 1.4, "y": 0.7},
                    {"name": "Delta Mus", "x": 1.8, "y": 1.2}
                ],
                "min_stars": 4,
                "min_confidence": 0.3,
                "description": "Musca - the fly"
            },
            "Puppis": {
                "name": "Puppis",
                "hemisphere": "southern",
                "shape_type": "ship",
                "pattern": [
                    {"from": 0, "to": 1, "distance_ratio": 1.0, "angle": 0},
                    {"from": 1, "to": 2, "distance_ratio": 0.9, "angle": 45},
                    {"from": 2, "to": 3, "distance_ratio": 0.8, "angle": 90}
                ],
                "star_positions": [
                    {"name": "Naos", "x": 0, "y": 0},
                    {"name": "Tureis", "x": 1, "y": 0},
                    {"name": "Asmidiske", "x": 1.4, "y": 0.7},
                    {"name": "HD 64760", "x": 1.7, "y": 1.4}
                ],
                "min_stars": 4,
                "min_confidence": 0.3,
                "description": "Puppis - the ship"
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
        """Find constellation patterns with improved shape matching."""
        logger.info("🔍 Searching for constellations with improved shape matching...")
        
        if not self.detected_stars:
            raise ValueError("No detected stars available")
        
        # Sort stars by brightness
        bright_stars = sorted(self.detected_stars, key=lambda s: s.get('brightness', 0), reverse=True)
        candidate_stars = bright_stars[:150]  # Use more stars for better coverage
        
        matches = []
        
        for const_name, const_data in self.constellation_shapes.items():
            logger.info(f"   Searching for {const_name}...")
            
            shape_matches = self._find_improved_shape_matches(candidate_stars, const_data)
            
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
                    "shape_score": match["shape_score"],
                    "pattern_connections": match["pattern_connections"]
                })
        
        # Apply improved spatial constraints
        constrained_matches = self._apply_improved_spatial_constraints(matches)
        
        # Sort by confidence
        constrained_matches.sort(key=lambda m: m["confidence"], reverse=True)
        
        logger.info(f"   Found {len(constrained_matches)} improved constellation matches")
        return constrained_matches
    
    def _find_improved_shape_matches(self, stars: List[Dict], const_data: Dict) -> List[Dict]:
        """Find matches for a specific constellation shape with improved pattern matching."""
        matches = []
        pattern = const_data["pattern"]
        min_stars = const_data["min_stars"]
        
        # Try different starting stars
        for start_star in stars[:30]:  # Try more starting points
            shape_matches = self._try_improved_pattern_from_star(start_star, stars, pattern, min_stars, const_data)
            matches.extend(shape_matches)
        
        return matches
    
    def _try_improved_pattern_from_star(self, start_star: Dict, all_stars: List[Dict], 
                                       pattern: List[Dict], min_stars: int, const_data: Dict) -> List[Dict]:
        """Try to match a constellation pattern with improved shape validation."""
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
            if base_distance < 60:  # Too close
                continue
            if base_distance > 500:  # Too far
                continue
            
            # Try to build the constellation pattern
            pattern_stars = [start_star, second_star]
            pattern_points = [(start_x, start_y), (second_star["x"], second_star["y"])]
            pattern_connections = []
            
            success = True
            total_score = 0
            shape_score = 0
            
            # Build the pattern step by step
            for pattern_step in pattern:
                from_idx = pattern_step["from"]
                to_idx = pattern_step["to"]
                expected_ratio = pattern_step["distance_ratio"]
                expected_angle = pattern_step["angle"]
                
                if from_idx >= len(pattern_stars):
                    success = False
                    break
                
                from_star = pattern_stars[from_idx]
                target_distance = base_distance * expected_ratio
                target_angle = base_angle + expected_angle
                
                # Find best matching star for the 'to' position
                best_star = None
                best_score = 0
                
                for candidate_star in all_stars:
                    if candidate_star in pattern_stars:
                        continue
                    
                    # Calculate distance and angle from 'from' star
                    dx = candidate_star["x"] - from_star["x"]
                    dy = candidate_star["y"] - from_star["y"]
                    actual_distance = math.sqrt(dx*dx + dy*dy)
                    actual_angle = math.degrees(math.atan2(dy, dx))
                    
                    # Calculate pattern matching score
                    distance_error = abs(actual_distance - target_distance) / target_distance
                    angle_error = abs(actual_angle - target_angle)
                    if angle_error > 180:
                        angle_error = 360 - angle_error
                    angle_error = angle_error / 180.0
                    
                    # Combined score with stricter tolerances
                    if distance_error < 0.3 and angle_error < 0.25:  # Tighter tolerances
                        score = 1.0 / (1.0 + distance_error + angle_error)
                        if score > best_score:
                            best_score = score
                            best_star = candidate_star
                
                if best_star:
                    pattern_stars.append(best_star)
                    pattern_points.append((best_star["x"], best_star["y"]))
                    pattern_connections.append((from_star, best_star))
                    total_score += best_score
                    shape_score += best_score
                else:
                    success = False
                    break
            
            if success and len(pattern_stars) >= min_stars:
                # Calculate final confidence
                confidence = total_score / len(pattern)
                
                # Add brightness bonus
                brightness_bonus = sum(s.get("brightness", 0) for s in pattern_stars) / (len(pattern_stars) * 255)
                confidence = 0.6 * confidence + 0.4 * brightness_bonus
                
                # Validate shape quality
                shape_quality = self._validate_shape_quality(pattern_stars, const_data)
                
                if confidence > 0.25 and shape_quality > 0.3:  # Lower threshold, but require shape quality
                    matches.append({
                        "confidence": confidence,
                        "stars": pattern_stars,
                        "pattern_points": pattern_points,
                        "shape_score": shape_quality,
                        "pattern_connections": pattern_connections
                    })
        
        return matches
    
    def _validate_shape_quality(self, stars: List[Dict], const_data: Dict) -> float:
        """Validate the quality of the constellation shape."""
        if len(stars) < 3:
            return 0.0
        
        # Calculate relative positions and compare to expected pattern
        star_positions = const_data.get("star_positions", [])
        if not star_positions:
            return 0.5  # Default score if no position data
        
        # Calculate center of detected stars
        center_x = np.mean([s["x"] for s in stars])
        center_y = np.mean([s["y"] for s in stars])
        
        # Normalize detected positions
        detected_positions = []
        for star in stars:
            dx = star["x"] - center_x
            dy = star["y"] - center_y
            distance = math.sqrt(dx*dx + dy*dy)
            if distance > 0:
                detected_positions.append({
                    "x": dx / distance,
                    "y": dy / distance
                })
        
        # Compare with expected pattern
        if len(detected_positions) >= len(star_positions):
            total_error = 0
            for i, expected in enumerate(star_positions[:len(detected_positions)]):
                if i < len(detected_positions):
                    detected = detected_positions[i]
                    error = math.sqrt((expected["x"] - detected["x"])**2 + (expected["y"] - detected["y"])**2)
                    total_error += error
            
            avg_error = total_error / len(star_positions)
            shape_quality = max(0, 1.0 - avg_error)
            return shape_quality
        
        return 0.3  # Default for insufficient stars
    
    def _apply_improved_spatial_constraints(self, matches: List[Dict]) -> List[Dict]:
        """Apply improved spatial constraints to allow more constellations."""
        logger.info("🌍 Applying improved spatial constraints...")
        
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
        
        # Rule 3: Shape quality validation (more lenient)
        valid_matches = []
        for match in matches:
            shape_score = match.get("shape_score", 0)
            if shape_score > 0.2:  # Lower threshold to allow more constellations
                valid_matches.append(match)
                logger.info(f"   ✅ Added {match['constellation_name']} (shape score: {shape_score:.2f})")
            else:
                logger.info(f"   ❌ Rejected {match['constellation_name']} (poor shape match: {shape_score:.2f})")
        
        # Rule 4: Spatial separation (more lenient)
        final_matches = []
        for match in valid_matches:
            is_valid = True
            for final_match in final_matches:
                distance = self._calculate_constellation_distance(match, final_match)
                if distance < 150:  # Reduced minimum separation
                    is_valid = False
                    logger.info(f"   ❌ Rejected {match['constellation_name']} (too close to {final_match['constellation_name']})")
                    break
            
            if is_valid:
                final_matches.append(match)
                logger.info(f"   ✅ Added {match['constellation_name']} (spatially separated)")
        
        # Rule 5: Increased FOV limits
        max_constellations = self._get_increased_max_constellations_for_fov()
        if len(final_matches) > max_constellations:
            logger.info(f"   Limiting to {max_constellations} constellations based on FOV")
            final_matches = final_matches[:max_constellations]
        
        return final_matches
    
    def _calculate_constellation_distance(self, match1: Dict, match2: Dict) -> float:
        """Calculate distance between two constellation centers."""
        dx = match1["center_x"] - match2["center_x"]
        dy = match1["center_y"] - match2["center_y"]
        return math.sqrt(dx*dx + dy*dy)
    
    def _get_increased_max_constellations_for_fov(self) -> int:
        """Get increased maximum number of constellations based on FOV."""
        fov_estimate = self.fov_info.get("fov_estimate", "unknown")
        
        if "narrow_field" in fov_estimate:
            return 4  # Increased from 2 to 4
        elif "medium_field" in fov_estimate:
            return 6  # Increased from 3 to 6
        else:
            return 8  # Increased from 4 to 8
    
    def create_annotated_image(self, matches: List[Dict], output_path: str = "Output/improved_shape_constellations.jpg"):
        """Create annotated image with improved constellation patterns."""
        logger.info("🎨 Creating improved shape annotated image...")
        
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
            self._draw_improved_constellation_pattern(annotated_image, match, color)
        
        # Add information overlay
        annotated_image = self._add_improved_info_overlay(annotated_image, matches)
        
        # Save result
        cv2.imwrite(output_path, annotated_image)
        logger.info(f"   ✅ Saved annotated image: {output_path}")
    
    def _draw_improved_constellation_pattern(self, image: np.ndarray, match: Dict, color: Tuple[int, int, int]):
        """Draw an improved constellation pattern on the image."""
        pattern_connections = match.get("pattern_connections", [])
        
        # Draw lines connecting the pattern based on actual connections
        for from_star, to_star in pattern_connections:
            pt1 = (int(from_star["x"]), int(from_star["y"]))
            pt2 = (int(to_star["x"]), int(to_star["y"]))
            cv2.line(image, pt1, pt2, color, 4)
        
        # Draw stars
        for star in match["matched_stars"]:
            cv2.circle(image, (int(star["x"]), int(star["y"])), 10, color, -1)
            cv2.circle(image, (int(star["x"]), int(star["y"])), 12, (255, 255, 255), 2)
        
        # Add constellation name, confidence, and shape score
        if match["matched_stars"]:
            text_pos = (int(match["matched_stars"][0]["x"]), int(match["matched_stars"][0]["y"]))
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
    
    def _add_improved_info_overlay(self, image: np.ndarray, matches: List[Dict]) -> np.ndarray:
        """Add improved information overlay to the image."""
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
        title = "Improved Shape Constellation Matching"
        draw.text((20, 20), title, fill=(255, 255, 255), font=font)
        
        # Add processing information
        y_offset = 80
        info_lines = [
            f"Stars analyzed: {len(self.detected_stars)}",
            f"Improved matches: {len(matches)}",
            f"FOV estimate: {self.fov_info.get('fov_estimate', 'unknown')}",
            f"Method: Improved pattern matching with shape validation"
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
        draw.text((20, y_offset), "Applied improvements:", 
                 fill=(255, 165, 0), font=small_font)
        y_offset += 25
        draw.text((20, y_offset), "• Proper constellation shapes", 
                 fill=(255, 165, 0), font=small_font)
        y_offset += 25
        draw.text((20, y_offset), "• More constellations allowed", 
                 fill=(255, 165, 0), font=small_font)
        y_offset += 25
        draw.text((20, y_offset), "• Better spatial orientation", 
                 fill=(255, 165, 0), font=small_font)
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def main():
    """Test improved shape constellation matching."""
    print("🔍 Improved Shape Constellation Matcher")
    print("=" * 60)
    print("Fixes shape matching and allows more constellations")
    print("=" * 60)
    
    start_time = time.time()
    
    # Create matcher
    matcher = ImprovedShapeConstellationMatcher()
    
    try:
        # Load processed data
        matcher.load_processed_data()
        
        # Find constellation patterns
        matches = matcher.find_constellation_patterns()
        
        # Create annotated image
        matcher.create_annotated_image(matches)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ Improved shape matching complete!")
        print(f"   Found {len(matches)} improved constellation matches")
        
        if matches:
            print("   Matched constellations:")
            for i, match in enumerate(matches):
                hemisphere = match["hemisphere"]
                shape_score = match.get("shape_score", 0)
                print(f"     {i+1}. {match['constellation_name']}: {match['confidence']:.2f} confidence ({hemisphere}, shape: {shape_score:.2f})")
        else:
            print("   No improved constellation patterns found")
            print("   This could indicate:")
            print("     - Image shows a different sky region")
            print("     - Need to adjust pattern matching parameters")
            print("     - Need more constellation patterns")
        
        print(f"   Total processing time: {elapsed_time:.1f} seconds")
        print(f"   Check Output/improved_shape_constellations.jpg for results")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        raise

if __name__ == "__main__":
    main() 