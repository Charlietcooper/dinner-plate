#!/usr/bin/env python3
"""
Galactic Center Star and Constellation Identifier
Designed for Sagittarius, Scorpius, and Corona Australis in dense star fields
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import json
import time
import argparse
from typing import List, Tuple, Dict, Optional, Set
import logging
from tqdm import tqdm
import math

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GalacticCenterIdentifier:
    """System designed for identifying stars and constellations in the galactic center."""
    
    def __init__(self):
        self.galactic_center_stars = self._create_galactic_center_stars()
        self.galactic_constellations = self._create_galactic_constellations()
        self.detected_stars = None
        self.fov_info = None
        self.identified_stars = []
        
    def _create_galactic_center_stars(self) -> Dict:
        """Create database of bright stars in the galactic center region."""
        return {
            # Sagittarius (The Archer) - Galactic Center
            "Kaus Australis": {
                "name": "Kaus Australis",
                "constellation": "Sagittarius",
                "magnitude": 1.85,
                "ra": 276.0430,
                "dec": -34.3847,
                "relationships": {
                    "Nunki": {"distance": 8.5, "angle": 45},
                    "Ascella": {"distance": 6.2, "angle": 90},
                    "Kaus Media": {"distance": 4.8, "angle": 0}
                },
                "description": "Epsilon Sagittarii - brightest star in Sagittarius"
            },
            "Nunki": {
                "name": "Nunki",
                "constellation": "Sagittarius",
                "magnitude": 2.05,
                "ra": 283.8164,
                "dec": -26.2967,
                "relationships": {
                    "Kaus Australis": {"distance": 8.5, "angle": 225},
                    "Ascella": {"distance": 3.8, "angle": 45},
                    "Kaus Media": {"distance": 7.2, "angle": 270}
                },
                "description": "Sigma Sagittarii - second brightest in Sagittarius"
            },
            "Ascella": {
                "name": "Ascella",
                "constellation": "Sagittarius",
                "magnitude": 2.60,
                "ra": 287.4409,
                "dec": -29.8281,
                "relationships": {
                    "Kaus Australis": {"distance": 6.2, "angle": 270},
                    "Nunki": {"distance": 3.8, "angle": 225},
                    "Kaus Media": {"distance": 4.1, "angle": 315}
                },
                "description": "Zeta Sagittarii - in the Teapot asterism"
            },
            "Kaus Media": {
                "name": "Kaus Media",
                "constellation": "Sagittarius",
                "magnitude": 2.72,
                "ra": 276.9929,
                "dec": -29.8281,
                "relationships": {
                    "Kaus Australis": {"distance": 4.8, "angle": 180},
                    "Nunki": {"distance": 7.2, "angle": 90},
                    "Ascella": {"distance": 4.1, "angle": 135}
                },
                "description": "Delta Sagittarii - in the Teapot asterism"
            },
            
            # Scorpius (The Scorpion)
            "Antares": {
                "name": "Antares",
                "constellation": "Scorpius",
                "magnitude": 1.06,
                "ra": 247.3519,
                "dec": -26.4320,
                "relationships": {
                    "Shaula": {"distance": 8.2, "angle": 180},
                    "Lesath": {"distance": 7.8, "angle": 200},
                    "Dschubba": {"distance": 6.5, "angle": 90}
                },
                "description": "Alpha Scorpii - red supergiant, heart of the scorpion"
            },
            "Shaula": {
                "name": "Shaula",
                "constellation": "Scorpius",
                "magnitude": 1.62,
                "ra": 263.4022,
                "dec": -37.1038,
                "relationships": {
                    "Antares": {"distance": 8.2, "angle": 0},
                    "Lesath": {"distance": 1.5, "angle": 20},
                    "Dschubba": {"distance": 5.8, "angle": 270}
                },
                "description": "Lambda Scorpii - stinger of the scorpion"
            },
            "Lesath": {
                "name": "Lesath",
                "constellation": "Scorpius",
                "magnitude": 2.70,
                "ra": 264.3297,
                "dec": -37.2953,
                "relationships": {
                    "Antares": {"distance": 7.8, "angle": 20},
                    "Shaula": {"distance": 1.5, "angle": 200},
                    "Dschubba": {"distance": 5.2, "angle": 290}
                },
                "description": "Upsilon Scorpii - stinger of the scorpion"
            },
            "Dschubba": {
                "name": "Dschubba",
                "constellation": "Scorpius",
                "magnitude": 2.29,
                "ra": 240.0833,
                "dec": -22.6217,
                "relationships": {
                    "Antares": {"distance": 6.5, "angle": 270},
                    "Shaula": {"distance": 5.8, "angle": 90},
                    "Lesath": {"distance": 5.2, "angle": 110}
                },
                "description": "Delta Scorpii - forehead of the scorpion"
            },
            
            # Corona Australis (The Southern Crown)
            "Alfecca Meridiana": {
                "name": "Alfecca Meridiana",
                "constellation": "Corona Australis",
                "magnitude": 4.10,
                "ra": 285.6530,
                "dec": -37.9045,
                "relationships": {
                    "Beta CrA": {"distance": 2.8, "angle": 45},
                    "Gamma CrA": {"distance": 3.2, "angle": 90},
                    "Delta CrA": {"distance": 4.1, "angle": 135}
                },
                "description": "Alpha Coronae Australis - brightest in the crown"
            },
            "Beta CrA": {
                "name": "Beta CrA",
                "constellation": "Corona Australis",
                "magnitude": 4.10,
                "ra": 286.1710,
                "dec": -37.9045,
                "relationships": {
                    "Alfecca Meridiana": {"distance": 2.8, "angle": 225},
                    "Gamma CrA": {"distance": 2.1, "angle": 45},
                    "Delta CrA": {"distance": 3.5, "angle": 90}
                },
                "description": "Beta Coronae Australis - in the crown"
            },
            
            # Additional bright stars in the region
            "Alnasl": {
                "name": "Alnasl",
                "constellation": "Sagittarius",
                "magnitude": 2.98,
                "ra": 271.4520,
                "dec": -30.4241,
                "relationships": {
                    "Kaus Australis": {"distance": 5.2, "angle": 315},
                    "Kaus Media": {"distance": 3.8, "angle": 45},
                    "Ascella": {"distance": 6.8, "angle": 90}
                },
                "description": "Gamma Sagittarii - tip of the arrow"
            },
            "Tau Sagittarii": {
                "name": "Tau Sagittarii",
                "constellation": "Sagittarius",
                "magnitude": 3.32,
                "ra": 285.6530,
                "dec": -27.6706,
                "relationships": {
                    "Nunki": {"distance": 2.8, "angle": 90},
                    "Ascella": {"distance": 1.8, "angle": 45},
                    "Kaus Media": {"distance": 5.2, "angle": 180}
                },
                "description": "Tau Sagittarii - in the Teapot"
            }
        }
    
    def _create_galactic_constellations(self) -> Dict:
        """Create constellation patterns for the galactic center region."""
        return {
            "Sagittarius": {
                "name": "Sagittarius",
                "hemisphere": "both",
                "anchor_stars": ["Kaus Australis", "Nunki", "Ascella", "Kaus Media"],
                "pattern": [
                    # Teapot asterism
                    {"from": "Kaus Australis", "to": "Kaus Media", "distance_ratio": 1.0, "angle": 0},
                    {"from": "Kaus Media", "to": "Ascella", "distance_ratio": 0.8, "angle": 45},
                    {"from": "Ascella", "to": "Nunki", "distance_ratio": 0.6, "angle": 45},
                    {"from": "Nunki", "to": "Kaus Australis", "distance_ratio": 1.2, "angle": 135}
                ],
                "min_stars": 4,
                "min_confidence": 0.5,
                "description": "Sagittarius - the Archer with Teapot asterism"
            },
            "Scorpius": {
                "name": "Scorpius",
                "hemisphere": "both",
                "anchor_stars": ["Antares", "Shaula", "Lesath", "Dschubba"],
                "pattern": [
                    # Scorpion body
                    {"from": "Antares", "to": "Dschubba", "distance_ratio": 1.0, "angle": 0},
                    {"from": "Dschubba", "to": "Shaula", "distance_ratio": 0.9, "angle": 90},
                    {"from": "Shaula", "to": "Lesath", "distance_ratio": 0.2, "angle": 20},
                    {"from": "Lesath", "to": "Antares", "distance_ratio": 1.1, "angle": 200}
                ],
                "min_stars": 4,
                "min_confidence": 0.5,
                "description": "Scorpius - the Scorpion"
            },
            "Corona Australis": {
                "name": "Corona Australis",
                "hemisphere": "southern",
                "anchor_stars": ["Alfecca Meridiana", "Beta CrA"],
                "pattern": [
                    # Crown shape
                    {"from": "Alfecca Meridiana", "to": "Beta CrA", "distance_ratio": 1.0, "angle": 0},
                    {"from": "Beta CrA", "to": "Gamma CrA", "distance_ratio": 0.8, "angle": 45},
                    {"from": "Gamma CrA", "to": "Delta CrA", "distance_ratio": 0.9, "angle": 45},
                    {"from": "Delta CrA", "to": "Alfecca Meridiana", "distance_ratio": 1.1, "angle": 225}
                ],
                "min_stars": 4,
                "min_confidence": 0.4,
                "description": "Corona Australis - the Southern Crown"
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
    
    def identify_galactic_stars(self) -> List[Dict]:
        """Identify bright stars in the galactic center region."""
        logger.info("⭐ Identifying stars in the galactic center...")
        
        if not self.detected_stars:
            raise ValueError("No detected stars available")
        
        # Sort stars by brightness
        bright_stars = sorted(self.detected_stars, key=lambda s: s.get('brightness', 0), reverse=True)
        candidate_stars = bright_stars[:100]  # More candidates for dense field
        
        identified_stars = []
        
        for star_name, star_data in self.galactic_center_stars.items():
            logger.info(f"   Searching for {star_name}...")
            
            # Find potential matches based on brightness
            potential_matches = self._find_star_by_brightness_dense_field(candidate_stars, star_data)
            
            for match in potential_matches:
                # Validate relationships with other identified stars
                relationship_score = self._validate_star_relationships_dense_field(match, star_data, identified_stars)
                
                if relationship_score > 0.5:  # Lower threshold for dense field
                    identified_stars.append({
                        "database_star": star_name,
                        "detected_star": match,
                        "confidence": relationship_score,
                        "magnitude": star_data["magnitude"],
                        "constellation": star_data["constellation"],
                        "description": star_data["description"]
                    })
                    logger.info(f"   ✅ Identified {star_name} with confidence {relationship_score:.2f}")
                    break
        
        self.identified_stars = identified_stars
        return identified_stars
    
    def _find_star_by_brightness_dense_field(self, candidate_stars: List[Dict], star_data: Dict) -> List[Dict]:
        """Find stars matching expected brightness in dense star field."""
        matches = []
        expected_magnitude = star_data["magnitude"]
        
        # Convert magnitude to expected brightness (inverse relationship)
        # Brighter stars have lower magnitude values
        expected_brightness = 255 * (1.0 / (1.0 + abs(expected_magnitude)))
        
        for star in candidate_stars:
            actual_brightness = star.get("brightness", 0)
            
            # Calculate brightness match score (more lenient for dense field)
            brightness_error = abs(actual_brightness - expected_brightness) / 255
            brightness_score = 1.0 / (1.0 + brightness_error)
            
            if brightness_score > 0.2:  # Lower threshold for dense field
                matches.append({
                    **star,
                    "brightness_score": brightness_score
                })
        
        # Sort by brightness score
        matches.sort(key=lambda m: m["brightness_score"], reverse=True)
        return matches[:5]  # Return top 5 matches for dense field
    
    def _validate_star_relationships_dense_field(self, candidate_star: Dict, star_data: Dict, identified_stars: List[Dict]) -> float:
        """Validate star relationships in dense star field."""
        if not identified_stars:
            return candidate_star.get("brightness_score", 0.4)  # Base score if no other stars
        
        relationships = star_data.get("relationships", {})
        total_score = 0
        valid_relationships = 0
        
        for identified_star in identified_stars:
            other_star_name = identified_star["database_star"]
            
            if other_star_name in relationships:
                expected_rel = relationships[other_star_name]
                actual_rel = self._calculate_star_relationship_dense_field(candidate_star, identified_star["detected_star"])
                
                # Compare distance and angle (more lenient for dense field)
                distance_error = abs(actual_rel["distance"] - expected_rel["distance"]) / expected_rel["distance"]
                angle_error = abs(actual_rel["angle"] - expected_rel["angle"])
                if angle_error > 180:
                    angle_error = 360 - angle_error
                angle_error = angle_error / 180.0
                
                # Combined score with more lenient tolerances
                if distance_error < 0.8 and angle_error < 0.5:  # More lenient for dense field
                    rel_score = 1.0 / (1.0 + distance_error + angle_error)
                    total_score += rel_score
                    valid_relationships += 1
        
        if valid_relationships == 0:
            return candidate_star.get("brightness_score", 0.4)
        
        avg_relationship_score = total_score / valid_relationships
        brightness_score = candidate_star.get("brightness_score", 0.4)
        
        # Combine relationship and brightness scores
        final_score = 0.6 * avg_relationship_score + 0.4 * brightness_score
        return final_score
    
    def _calculate_star_relationship_dense_field(self, star1: Dict, star2: Dict) -> Dict:
        """Calculate distance and angle between stars in dense field."""
        dx = star2["x"] - star1["x"]
        dy = star2["y"] - star1["y"]
        
        # Convert pixel distance to angular distance (approximate)
        # For dense galactic center field, use smaller scale
        pixel_to_degree = 0.001  # Smaller scale for dense field
        distance = math.sqrt(dx*dx + dy*dy) * pixel_to_degree
        
        angle = math.degrees(math.atan2(dy, dx))
        if angle < 0:
            angle += 360
        
        return {
            "distance": distance,
            "angle": angle
        }
    
    def find_galactic_constellations(self) -> List[Dict]:
        """Find constellations in the galactic center using identified stars as anchors."""
        logger.info("🔍 Finding constellations in the galactic center...")
        
        if not self.identified_stars:
            logger.warning("   No identified stars available - running star identification first")
            self.identify_galactic_stars()
        
        matches = []
        
        for const_name, const_data in self.galactic_constellations.items():
            logger.info(f"   Searching for {const_name} using anchor stars...")
            
            # Check if we have the required anchor stars
            anchor_stars = const_data["anchor_stars"]
            available_anchors = [s for s in self.identified_stars if s["database_star"] in anchor_stars]
            
            if len(available_anchors) >= 1:  # Need at least one anchor
                constellation_matches = self._find_constellation_with_anchors_dense_field(const_data, available_anchors)
                matches.extend(constellation_matches)
            else:
                logger.info(f"   ❌ No anchor stars available for {const_name}")
        
        # Apply spatial constraints
        constrained_matches = self._apply_galactic_constraints(matches)
        
        # Sort by confidence
        constrained_matches.sort(key=lambda m: m["confidence"], reverse=True)
        
        logger.info(f"   Found {len(constrained_matches)} constellation matches in galactic center")
        return constrained_matches
    
    def _find_constellation_with_anchors_dense_field(self, const_data: Dict, anchor_stars: List[Dict]) -> List[Dict]:
        """Find constellation pattern using identified stars as anchors in dense field."""
        matches = []
        pattern = const_data["pattern"]
        
        # Use the first anchor star as the starting point
        anchor_star = anchor_stars[0]
        anchor_detected = anchor_star["detected_star"]
        
        # Build constellation pattern from anchor
        pattern_stars = [anchor_detected]
        pattern_points = [(anchor_detected["x"], anchor_detected["y"])]
        
        success = True
        total_score = anchor_star["confidence"]
        
        for pattern_step in pattern:
            from_star_name = pattern_step["from"]
            to_star_name = pattern_step["to"]
            expected_ratio = pattern_step["distance_ratio"]
            expected_angle = pattern_step["angle"]
            
            # Find the 'from' star (could be anchor or previously found star)
            from_star = None
            if from_star_name == anchor_star["database_star"]:
                from_star = anchor_detected
            else:
                # Find in pattern_stars by matching to identified stars
                for i, star in enumerate(pattern_stars):
                    if i < len(pattern_stars) - 1:  # Not the last one
                        from_star = star
                        break
            
            if not from_star:
                success = False
                break
            
            # Find the 'to' star
            target_star = self._find_target_star_dense_field(from_star, expected_ratio, expected_angle, pattern_stars)
            
            if target_star:
                pattern_stars.append(target_star)
                pattern_points.append((target_star["x"], target_star["y"]))
                total_score += 0.8  # Good pattern match
            else:
                success = False
                break
        
        if success and len(pattern_stars) >= const_data["min_stars"]:
            confidence = total_score / len(pattern)
            
            if confidence >= const_data["min_confidence"]:
                matches.append({
                    "constellation": const_data["name"],
                    "constellation_name": const_data["name"],
                    "hemisphere": const_data["hemisphere"],
                    "description": const_data["description"],
                    "confidence": confidence,
                    "matched_stars": pattern_stars,
                    "pattern_points": pattern_points,
                    "center_x": np.mean([s["x"] for s in pattern_stars]),
                    "center_y": np.mean([s["y"] for s in pattern_stars]),
                    "anchor_stars_used": [s["database_star"] for s in anchor_stars]
                })
        
        return matches
    
    def _find_target_star_dense_field(self, from_star: Dict, expected_ratio: float, expected_angle: float, 
                                     existing_stars: List[Dict]) -> Optional[Dict]:
        """Find a star at the expected distance and angle in dense field."""
        base_distance = 80  # pixels (smaller for dense field)
        target_distance = base_distance * expected_ratio
        target_angle = math.radians(expected_angle)
        
        best_star = None
        best_score = 0
        
        for star in self.detected_stars:
            if star in existing_stars:
                continue
            
            # Calculate distance and angle from 'from_star'
            dx = star["x"] - from_star["x"]
            dy = star["y"] - from_star["y"]
            actual_distance = math.sqrt(dx*dx + dy*dy)
            actual_angle = math.atan2(dy, dx)
            
            # Calculate match score (more lenient for dense field)
            distance_error = abs(actual_distance - target_distance) / target_distance
            angle_error = abs(actual_angle - target_angle)
            if angle_error > math.pi:
                angle_error = 2 * math.pi - angle_error
            angle_error = angle_error / math.pi
            
            if distance_error < 0.6 and angle_error < 0.4:  # More lenient tolerances
                score = 1.0 / (1.0 + distance_error + angle_error)
                if score > best_score:
                    best_score = score
                    best_star = star
        
        return best_star
    
    def _apply_galactic_constraints(self, matches: List[Dict]) -> List[Dict]:
        """Apply spatial constraints for galactic center constellations."""
        logger.info("🌍 Applying galactic center constraints...")
        
        if not matches:
            return []
        
        # Remove duplicates (keep highest confidence)
        unique_matches = {}
        for match in matches:
            const_name = match["constellation"]
            if const_name not in unique_matches or match["confidence"] > unique_matches[const_name]["confidence"]:
                unique_matches[const_name] = match
        
        matches = list(unique_matches.values())
        
        # Spatial separation (smaller for dense field)
        final_matches = []
        for match in matches:
            is_valid = True
            for final_match in final_matches:
                distance = self._calculate_constellation_distance(match, final_match)
                if distance < 150:  # Smaller minimum separation for dense field
                    is_valid = False
                    break
            
            if is_valid:
                final_matches.append(match)
        
        return final_matches
    
    def _calculate_constellation_distance(self, match1: Dict, match2: Dict) -> float:
        """Calculate distance between two constellation centers."""
        dx = match1["center_x"] - match2["center_x"]
        dy = match1["center_y"] - match2["center_y"]
        return math.sqrt(dx*dx + dy*dy)
    
    def create_annotated_image(self, mode: str, output_path: str = None):
        """Create annotated image based on mode."""
        if mode == "stars":
            self._create_galactic_star_annotated_image(output_path or "Output/galactic_stars.jpg")
        elif mode == "constellations":
            self._create_galactic_constellation_annotated_image(output_path or "Output/galactic_constellations.jpg")
        else:
            raise ValueError("Mode must be 'stars' or 'constellations'")
    
    def _create_galactic_star_annotated_image(self, output_path: str):
        """Create annotated image showing identified stars in galactic center."""
        logger.info("🎨 Creating galactic center star identification image...")
        
        # Load original image
        original_image = cv2.imread("Input/test-1.jpg")
        if original_image is None:
            logger.error("❌ Could not load original image")
            return
        
        # Create annotation overlay
        annotated_image = original_image.copy()
        
        # Draw identified stars
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        for i, identified_star in enumerate(self.identified_stars):
            color = colors[i % len(colors)]
            star = identified_star["detected_star"]
            
            # Draw star
            cv2.circle(annotated_image, (int(star["x"]), int(star["y"])), 12, color, -1)
            cv2.circle(annotated_image, (int(star["x"]), int(star["y"])), 15, (255, 255, 255), 3)
            
            # Add star name and magnitude
            label = f"{identified_star['database_star']} (mag {identified_star['magnitude']})"
            cv2.putText(annotated_image, label,
                       (int(star["x"]) + 18, int(star["y"]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Add confidence and constellation
            conf_label = f"Conf: {identified_star['confidence']:.2f} ({identified_star['constellation']})"
            cv2.putText(annotated_image, conf_label,
                       (int(star["x"]) + 18, int(star["y"]) + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add information overlay
        annotated_image = self._add_galactic_info_overlay(annotated_image)
        
        # Save result
        cv2.imwrite(output_path, annotated_image)
        logger.info(f"   ✅ Saved galactic star identification image: {output_path}")
    
    def _create_galactic_constellation_annotated_image(self, output_path: str):
        """Create annotated image showing constellations in galactic center."""
        logger.info("🎨 Creating galactic constellation annotation image...")
        
        # Load original image
        original_image = cv2.imread("Input/test-1.jpg")
        if original_image is None:
            logger.error("❌ Could not load original image")
            return
        
        # Create annotation overlay
        annotated_image = original_image.copy()
        
        # Draw identified stars first (as anchors)
        for identified_star in self.identified_stars:
            star = identified_star["detected_star"]
            cv2.circle(annotated_image, (int(star["x"]), int(star["y"])), 10, (255, 255, 255), -1)
            cv2.circle(annotated_image, (int(star["x"]), int(star["y"])), 13, (255, 255, 255), 2)
        
        # Draw constellations
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        # Get constellation matches
        constellation_matches = self.find_galactic_constellations()
        
        for i, match in enumerate(constellation_matches):
            color = colors[i % len(colors)]
            pattern_points = match["pattern_points"]
            
            # Draw constellation lines
            for j in range(len(pattern_points) - 1):
                pt1 = tuple(map(int, pattern_points[j]))
                pt2 = tuple(map(int, pattern_points[j + 1]))
                cv2.line(annotated_image, pt1, pt2, color, 3)
            
            # Draw constellation stars
            for point in pattern_points:
                cv2.circle(annotated_image, tuple(map(int, point)), 6, color, -1)
            
            # Add constellation name and confidence
            if pattern_points:
                text_pos = tuple(map(int, pattern_points[0]))
                label = f"{match['constellation_name']} ({match['confidence']:.2f})"
                cv2.putText(annotated_image, label,
                           (text_pos[0] + 12, text_pos[1] - 12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Add anchor stars info
                anchors_label = f"Anchors: {', '.join(match['anchor_stars_used'])}"
                cv2.putText(annotated_image, anchors_label,
                           (text_pos[0] + 12, text_pos[1] + 12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add information overlay
        annotated_image = self._add_galactic_constellation_info_overlay(annotated_image, constellation_matches)
        
        # Save result
        cv2.imwrite(output_path, annotated_image)
        logger.info(f"   ✅ Saved galactic constellation image: {output_path}")
    
    def _add_galactic_info_overlay(self, image: np.ndarray) -> np.ndarray:
        """Add information overlay for galactic center star identification."""
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
        title = "Galactic Center Star Identification"
        draw.text((20, 20), title, fill=(255, 255, 255), font=font)
        
        # Add processing information
        y_offset = 80
        info_lines = [
            f"Stars analyzed: {len(self.detected_stars)}",
            f"Galactic stars identified: {len(self.identified_stars)}",
            f"FOV estimate: {self.fov_info.get('fov_estimate', 'unknown')}",
            f"Method: Dense field magnitude and relationship matching"
        ]
        
        for line in info_lines:
            draw.text((20, y_offset), line, fill=(255, 255, 255), font=small_font)
            y_offset += 35
        
        # Add identified star details
        y_offset += 20
        for i, identified_star in enumerate(self.identified_stars):
            star_name = identified_star["database_star"]
            magnitude = identified_star["magnitude"]
            confidence = identified_star["confidence"]
            constellation = identified_star["constellation"]
            
            line = f"• {star_name}: mag {magnitude}, conf {confidence:.2f} ({constellation})"
            draw.text((20, y_offset), line, fill=(0, 255, 255), font=small_font)
            y_offset += 30
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def _add_galactic_constellation_info_overlay(self, image: np.ndarray, constellation_matches: List[Dict]) -> np.ndarray:
        """Add information overlay for galactic center constellation identification."""
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
        title = "Galactic Center Constellation Identification"
        draw.text((20, 20), title, fill=(255, 255, 255), font=font)
        
        # Add processing information
        y_offset = 80
        info_lines = [
            f"Stars analyzed: {len(self.detected_stars)}",
            f"Anchor stars identified: {len(self.identified_stars)}",
            f"Constellations found: {len(constellation_matches)}",
            f"Method: Dense field anchor-guided pattern matching"
        ]
        
        for line in info_lines:
            draw.text((20, y_offset), line, fill=(255, 255, 255), font=small_font)
            y_offset += 35
        
        # Add constellation details
        y_offset += 20
        for i, match in enumerate(constellation_matches):
            const_name = match["constellation_name"]
            confidence = match["confidence"]
            anchors = ", ".join(match["anchor_stars_used"])
            num_stars = len(match["matched_stars"])
            
            line = f"• {const_name}: {confidence:.2f} confidence ({num_stars} stars)"
            draw.text((20, y_offset), line, fill=(0, 255, 255), font=small_font)
            y_offset += 25
            line2 = f"  Anchors: {anchors}"
            draw.text((20, y_offset), line2, fill=(255, 165, 0), font=small_font)
            y_offset += 30
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(description="Galactic Center Star and Constellation Identifier")
    parser.add_argument("mode", choices=["stars", "constellations"], 
                       help="Mode: 'stars' to identify bright stars, 'constellations' to find constellations")
    parser.add_argument("--output", "-o", type=str, 
                       help="Output file path (optional)")
    
    args = parser.parse_args()
    
    print(f"🔍 Galactic Center Identifier - {args.mode.upper()} mode")
    print("=" * 60)
    print("Designed for Sagittarius, Scorpius, and Corona Australis")
    print("=" * 60)
    
    start_time = time.time()
    
    # Create identifier
    identifier = GalacticCenterIdentifier()
    
    try:
        # Load processed data
        identifier.load_processed_data()
        
        if args.mode == "stars":
            # Identify bright stars in galactic center
            identified_stars = identifier.identify_galactic_stars()
            
            print(f"\n✅ Galactic center star identification complete!")
            print(f"   Found {len(identified_stars)} bright stars")
            
            if identified_stars:
                print("   Identified stars:")
                for i, star in enumerate(identified_stars):
                    print(f"     {i+1}. {star['database_star']}: mag {star['magnitude']}, conf {star['confidence']:.2f} ({star['constellation']})")
            else:
                print("   No bright stars identified")
                print("   This could indicate:")
                print("     - Very dense star field making identification difficult")
                print("     - Need to adjust magnitude thresholds")
                print("     - Need to expand bright star database")
        
        elif args.mode == "constellations":
            # First identify stars, then find constellations
            identified_stars = identifier.identify_galactic_stars()
            constellation_matches = identifier.find_galactic_constellations()
            
            print(f"\n✅ Galactic center constellation identification complete!")
            print(f"   Found {len(identified_stars)} anchor stars")
            print(f"   Found {len(constellation_matches)} constellations")
            
            if constellation_matches:
                print("   Identified constellations:")
                for i, match in enumerate(constellation_matches):
                    anchors = ", ".join(match["anchor_stars_used"])
                    print(f"     {i+1}. {match['constellation_name']}: {match['confidence']:.2f} confidence (anchors: {anchors})")
            else:
                print("   No constellations identified")
                print("   This could indicate:")
                print("     - Need more anchor stars")
                print("     - Very dense field making pattern matching difficult")
                print("     - Need to adjust pattern matching parameters")
        
        # Create annotated image
        identifier.create_annotated_image(args.mode, args.output)
        
        elapsed_time = time.time() - start_time
        print(f"   Total processing time: {elapsed_time:.1f} seconds")
        
        if args.output:
            print(f"   Check {args.output} for results")
        else:
            if args.mode == "stars":
                print(f"   Check Output/galactic_stars.jpg for results")
            else:
                print(f"   Check Output/galactic_constellations.jpg for results")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        raise

if __name__ == "__main__":
    main() 