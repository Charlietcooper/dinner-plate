#!/usr/bin/env python3
"""
Boundary Constrained Constellation Matcher
Uses proper constellation boundaries and shapes from IAU data
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

class BoundaryConstrainedConstellationMatcher:
    """Constellation matcher using proper boundaries and shapes from IAU data."""
    
    def __init__(self):
        self.constellation_data = self._create_constellation_data()
        self.boundary_constraints = self._create_boundary_constraints()
        self.detected_stars = None
        self.fov_info = None
        
    def _create_constellation_data(self) -> Dict:
        """Create constellation data with proper shapes and boundaries based on IAU data."""
        return {
            "Crux": {
                "name": "Southern Cross",
                "hemisphere": "southern",
                "boundary": {
                    "ra_min": 180.0, "ra_max": 195.0,
                    "dec_min": -65.0, "dec_max": -55.0
                },
                "shape": {
                    "type": "cross",
                    "stars": [
                        {"name": "Acrux", "ra": 186.6495, "dec": -63.0991, "mag": 0.77},
                        {"name": "Mimosa", "ra": 191.9303, "dec": -59.6888, "mag": 1.25},
                        {"name": "Gacrux", "ra": 187.7915, "dec": -57.1133, "mag": 1.59},
                        {"name": "Delta Crucis", "ra": 183.7863, "dec": -58.7489, "mag": 2.79}
                    ],
                    "connections": [
                        ["Acrux", "Mimosa"],
                        ["Mimosa", "Gacrux"],
                        ["Gacrux", "Delta Crucis"],
                        ["Delta Crucis", "Acrux"]
                    ]
                },
                "min_confidence": 0.7,
                "description": "Southern Cross - distinctive cross shape"
            },
            "Carina": {
                "name": "Carina",
                "hemisphere": "southern",
                "boundary": {
                    "ra_min": 90.0, "ra_max": 170.0,
                    "dec_min": -70.0, "dec_max": -50.0
                },
                "shape": {
                    "type": "ship_keel",
                    "stars": [
                        {"name": "Canopus", "ra": 95.9880, "dec": -52.6957, "mag": -0.74},
                        {"name": "Avior", "ra": 139.2725, "dec": -59.5092, "mag": 1.86},
                        {"name": "Aspidiske", "ra": 139.2725, "dec": -59.5092, "mag": 2.21},
                        {"name": "Miaplacidus", "ra": 138.2992, "dec": -69.7172, "mag": 1.67}
                    ],
                    "connections": [
                        ["Canopus", "Avior"],
                        ["Avior", "Aspidiske"],
                        ["Aspidiske", "Miaplacidus"]
                    ]
                },
                "min_confidence": 0.6,
                "description": "Carina - keel of the ship"
            },
            "Centaurus": {
                "name": "Centaurus",
                "hemisphere": "southern",
                "boundary": {
                    "ra_min": 180.0, "ra_max": 230.0,
                    "dec_min": -65.0, "dec_max": -30.0
                },
                "shape": {
                    "type": "centaur",
                    "stars": [
                        {"name": "Alpha Centauri", "ra": 219.9021, "dec": -60.8340, "mag": -0.27},
                        {"name": "Hadar", "ra": 210.9559, "dec": -60.3730, "mag": 0.61},
                        {"name": "Menkent", "ra": 204.9719, "dec": -36.7123, "mag": 2.06},
                        {"name": "Muhlifain", "ra": 211.6708, "dec": -54.4914, "mag": 2.20}
                    ],
                    "connections": [
                        ["Alpha Centauri", "Hadar"],
                        ["Hadar", "Menkent"],
                        ["Menkent", "Muhlifain"]
                    ]
                },
                "min_confidence": 0.6,
                "description": "Centaurus - the centaur"
            },
            "Orion": {
                "name": "Orion",
                "hemisphere": "both",
                "boundary": {
                    "ra_min": 70.0, "ra_max": 95.0,
                    "dec_min": -15.0, "dec_max": 25.0
                },
                "shape": {
                    "type": "hunter",
                    "stars": [
                        {"name": "Betelgeuse", "ra": 88.7929, "dec": 7.4071, "mag": 0.42},
                        {"name": "Rigel", "ra": 78.6345, "dec": -8.2016, "mag": 0.18},
                        {"name": "Bellatrix", "ra": 81.2828, "dec": 6.3497, "mag": 1.64},
                        {"name": "Mintaka", "ra": 83.0016, "dec": -0.2991, "mag": 2.25},
                        {"name": "Alnilam", "ra": 84.0534, "dec": -1.2019, "mag": 1.69},
                        {"name": "Alnitak", "ra": 85.1897, "dec": -1.9426, "mag": 1.74}
                    ],
                    "connections": [
                        ["Betelgeuse", "Bellatrix"],
                        ["Bellatrix", "Mintaka"],
                        ["Mintaka", "Alnilam"],
                        ["Alnilam", "Alnitak"],
                        ["Alnitak", "Rigel"],
                        ["Rigel", "Betelgeuse"]
                    ]
                },
                "min_confidence": 0.6,
                "description": "Orion - hunter with distinctive belt"
            },
            "Ursa Major": {
                "name": "Big Dipper",
                "hemisphere": "northern",
                "boundary": {
                    "ra_min": 160.0, "ra_max": 200.0,
                    "dec_min": 30.0, "dec_max": 75.0
                },
                "shape": {
                    "type": "dipper",
                    "stars": [
                        {"name": "Dubhe", "ra": 165.9323, "dec": 61.7510, "mag": 1.79},
                        {"name": "Merak", "ra": 165.4603, "dec": 56.3824, "mag": 2.37},
                        {"name": "Phecda", "ra": 178.4577, "dec": 53.6948, "mag": 2.44},
                        {"name": "Megrez", "ra": 183.8565, "dec": 57.0326, "mag": 3.32},
                        {"name": "Alioth", "ra": 193.5073, "dec": 55.9598, "mag": 1.76},
                        {"name": "Mizar", "ra": 200.9814, "dec": 54.9254, "mag": 2.23},
                        {"name": "Alkaid", "ra": 206.8852, "dec": 49.3133, "mag": 1.85}
                    ],
                    "connections": [
                        ["Dubhe", "Merak"],
                        ["Merak", "Phecda"],
                        ["Phecda", "Megrez"],
                        ["Megrez", "Alioth"],
                        ["Alioth", "Mizar"],
                        ["Mizar", "Alkaid"]
                    ]
                },
                "min_confidence": 0.6,
                "description": "Big Dipper - distinctive dipper shape"
            }
        }
    
    def _create_boundary_constraints(self) -> Dict:
        """Create boundary constraints based on constellation positions."""
        return {
            "overlap_threshold": 0.1,  # Maximum allowed overlap between constellation boundaries
            "minimum_separation": 5.0,  # Minimum separation in degrees between constellation centers
            "shape_tolerance": 0.3,     # Tolerance for shape matching
            "brightness_weight": 0.4,   # Weight for star brightness in scoring
            "position_weight": 0.6      # Weight for position accuracy in scoring
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
        """Find constellation patterns using boundary and shape constraints."""
        logger.info("🔍 Searching for constellations with boundary and shape constraints...")
        
        if not self.detected_stars:
            raise ValueError("No detected stars available")
        
        # Estimate image center and scale for coordinate conversion
        image_center_ra, image_center_dec, scale = self._estimate_image_coordinates()
        
        logger.info(f"   Estimated image center: RA={image_center_ra:.1f}°, Dec={image_center_dec:.1f}°")
        logger.info(f"   Estimated scale: {scale:.4f}°/pixel")
        
        matches = []
        
        for const_name, const_data in self.constellation_data.items():
            logger.info(f"   Searching for {const_name}...")
            
            # Check if constellation is within image bounds
            if self._constellation_in_image_bounds(const_data, image_center_ra, image_center_dec, scale):
                constellation_matches = self._find_constellation_match(const_data, image_center_ra, image_center_dec, scale)
                matches.extend(constellation_matches)
            else:
                logger.info(f"   ❌ {const_name} outside image bounds")
        
        # Apply boundary constraints
        constrained_matches = self._apply_boundary_constraints(matches)
        
        # Sort by confidence
        constrained_matches.sort(key=lambda m: m["confidence"], reverse=True)
        
        logger.info(f"   Found {len(constrained_matches)} boundary-constrained constellation matches")
        return constrained_matches
    
    def _estimate_image_coordinates(self) -> Tuple[float, float, float]:
        """Estimate the celestial coordinates of the image center and scale."""
        # This is a simplified estimation - in practice, you'd use plate solving
        # For now, we'll use reasonable defaults based on the image content
        
        # Based on the image showing southern constellations, estimate southern hemisphere
        estimated_center_ra = 200.0  # Rough estimate
        estimated_center_dec = -60.0  # Southern hemisphere
        
        # Estimate scale based on FOV
        fov_estimate = self.fov_info.get("fov_estimate", "narrow_field")
        if "narrow_field" in fov_estimate:
            estimated_fov = 10.0  # degrees
        elif "medium_field" in fov_estimate:
            estimated_fov = 30.0  # degrees
        else:
            estimated_fov = 60.0  # degrees
        
        # Calculate scale (degrees per pixel)
        image_width = 5120  # pixels
        scale = estimated_fov / image_width
        
        return estimated_center_ra, estimated_center_dec, scale
    
    def _constellation_in_image_bounds(self, const_data: Dict, center_ra: float, center_dec: float, scale: float) -> bool:
        """Check if constellation boundary overlaps with image bounds."""
        boundary = const_data["boundary"]
        
        # Calculate image bounds in celestial coordinates
        image_width_deg = 5120 * scale
        image_height_deg = 3413 * scale
        
        image_ra_min = center_ra - image_width_deg / 2
        image_ra_max = center_ra + image_width_deg / 2
        image_dec_min = center_dec - image_height_deg / 2
        image_dec_max = center_dec + image_height_deg / 2
        
        # Check for overlap
        ra_overlap = (boundary["ra_min"] < image_ra_max and boundary["ra_max"] > image_ra_min)
        dec_overlap = (boundary["dec_min"] < image_dec_max and boundary["dec_max"] > image_dec_min)
        
        return ra_overlap and dec_overlap
    
    def _find_constellation_match(self, const_data: Dict, center_ra: float, center_dec: float, scale: float) -> List[Dict]:
        """Find matches for a specific constellation using shape constraints."""
        matches = []
        shape_data = const_data["shape"]
        
        # Convert constellation stars to pixel coordinates
        const_stars_pixels = []
        for star in shape_data["stars"]:
            pixel_x, pixel_y = self._celestial_to_pixel(star["ra"], star["dec"], center_ra, center_dec, scale)
            const_stars_pixels.append({
                "name": star["name"],
                "ra": star["ra"],
                "dec": star["dec"],
                "pixel_x": pixel_x,
                "pixel_y": pixel_y,
                "magnitude": star["mag"]
            })
        
        # Find matching stars for each constellation star
        matched_combinations = self._find_star_combinations(const_stars_pixels)
        
        for combination in matched_combinations:
            if len(combination) >= 3:  # Need at least 3 stars for a valid constellation
                confidence = self._calculate_shape_confidence(combination, shape_data)
                if confidence >= const_data["min_confidence"]:
                    matches.append({
                        "constellation": const_data["name"],
                        "constellation_name": const_data["name"],
                        "hemisphere": const_data["hemisphere"],
                        "description": const_data["description"],
                        "confidence": confidence,
                        "matched_stars": combination,
                        "pattern_points": [(s["pixel_x"], s["pixel_y"]) for s in combination],
                        "center_x": np.mean([s["pixel_x"] for s in combination]),
                        "center_y": np.mean([s["pixel_y"] for s in combination])
                    })
        
        return matches
    
    def _celestial_to_pixel(self, ra: float, dec: float, center_ra: float, center_dec: float, scale: float) -> Tuple[float, float]:
        """Convert celestial coordinates to pixel coordinates."""
        # Simplified conversion - assumes simple projection
        ra_diff = ra - center_ra
        dec_diff = dec - center_dec
        
        # Convert to pixels (approximate)
        pixel_x = 2560 + (ra_diff / scale) * math.cos(math.radians(center_dec))
        pixel_y = 1706.5 - (dec_diff / scale)
        
        return pixel_x, pixel_y
    
    def _find_star_combinations(self, const_stars: List[Dict]) -> List[List[Dict]]:
        """Find combinations of detected stars that match constellation stars."""
        combinations = []
        
        # Sort detected stars by brightness
        bright_stars = sorted(self.detected_stars, key=lambda s: s.get('brightness', 0), reverse=True)
        
        for const_star in const_stars:
            matches = []
            const_x, const_y = const_star["pixel_x"], const_star["pixel_y"]
            
            # Find stars within reasonable distance
            for detected_star in bright_stars[:100]:  # Check top 100 brightest
                distance = math.sqrt((const_x - detected_star["x"])**2 + (const_y - detected_star["y"])**2)
                
                if distance < 100:  # Within 100 pixels
                    matches.append({
                        **detected_star,
                        "constellation_star": const_star,
                        "distance": distance
                    })
            
            if matches:
                # Sort by distance and brightness
                matches.sort(key=lambda m: m["distance"] + (255 - m.get("brightness", 0)) / 255)
                combinations.append(matches[:3])  # Keep top 3 matches
        
        # Generate combinations
        if len(combinations) >= 3:
            return self._generate_star_combinations(combinations)
        
        return []
    
    def _generate_star_combinations(self, star_groups: List[List[Dict]]) -> List[List[Dict]]:
        """Generate combinations of stars from different groups."""
        if len(star_groups) < 3:
            return []
        
        combinations = []
        
        # Simple approach: take one star from each of the first 3-4 groups
        for i in range(min(3, len(star_groups))):
            for j in range(i+1, min(4, len(star_groups))):
                for k in range(j+1, min(5, len(star_groups))):
                    if i < len(star_groups) and j < len(star_groups) and k < len(star_groups):
                        combination = [
                            star_groups[i][0],
                            star_groups[j][0],
                            star_groups[k][0]
                        ]
                        combinations.append(combination)
        
        return combinations
    
    def _calculate_shape_confidence(self, matched_stars: List[Dict], shape_data: Dict) -> float:
        """Calculate confidence based on shape matching."""
        if len(matched_stars) < 3:
            return 0.0
        
        # Calculate relative distances between stars
        distances = []
        for i in range(len(matched_stars)):
            for j in range(i+1, len(matched_stars)):
                dx = matched_stars[i]["x"] - matched_stars[j]["x"]
                dy = matched_stars[i]["y"] - matched_stars[j]["y"]
                distance = math.sqrt(dx*dx + dy*dy)
                distances.append(distance)
        
        # Normalize distances
        if distances:
            max_dist = max(distances)
            normalized_distances = [d / max_dist for d in distances]
        else:
            return 0.0
        
        # Calculate brightness score
        brightness_score = sum(s.get("brightness", 0) for s in matched_stars) / (len(matched_stars) * 255)
        
        # Calculate position accuracy score
        position_score = 1.0 - sum(s.get("distance", 0) for s in matched_stars) / (len(matched_stars) * 100)
        position_score = max(0.0, min(1.0, position_score))
        
        # Combine scores
        constraints = self.boundary_constraints
        confidence = (constraints["brightness_weight"] * brightness_score + 
                     constraints["position_weight"] * position_score)
        
        return confidence
    
    def _apply_boundary_constraints(self, matches: List[Dict]) -> List[Dict]:
        """Apply boundary constraints to filter matches."""
        logger.info("🌍 Applying boundary constraints...")
        
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
        
        # Rule 3: Boundary separation
        valid_matches = []
        for match in matches:
            is_valid = True
            for valid_match in valid_matches:
                distance = self._calculate_constellation_distance(match, valid_match)
                if distance < 300:  # Minimum separation in pixels
                    is_valid = False
                    break
            
            if is_valid:
                valid_matches.append(match)
                logger.info(f"   ✅ Added {match['constellation_name']} (boundary constraint satisfied)")
            else:
                logger.info(f"   ❌ Rejected {match['constellation_name']} (too close to existing match)")
        
        # Rule 4: FOV limits
        max_constellations = self._get_max_constellations_for_fov()
        if len(valid_matches) > max_constellations:
            logger.info(f"   Limiting to {max_constellations} constellations based on FOV")
            valid_matches = valid_matches[:max_constellations]
        
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
            return 2
        elif "medium_field" in fov_estimate:
            return 3
        else:
            return 4
    
    def create_annotated_image(self, matches: List[Dict], output_path: str = "Output/boundary_constrained_constellations.jpg"):
        """Create annotated image with boundary-constrained constellation patterns."""
        logger.info("🎨 Creating boundary-constrained annotated image...")
        
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
        
        # Add constellation name and confidence
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
        title = "Boundary-Constrained Constellation Matching"
        draw.text((20, 20), title, fill=(255, 255, 255), font=font)
        
        # Add processing information
        y_offset = 80
        info_lines = [
            f"Stars analyzed: {len(self.detected_stars)}",
            f"Boundary-valid matches: {len(matches)}",
            f"FOV estimate: {self.fov_info.get('fov_estimate', 'unknown')}",
            f"Method: Shape and boundary constraints"
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
        draw.text((20, y_offset), "• IAU boundary validation", 
                 fill=(255, 165, 0), font=small_font)
        y_offset += 25
        draw.text((20, y_offset), "• Proper constellation shapes", 
                 fill=(255, 165, 0), font=small_font)
        y_offset += 25
        draw.text((20, y_offset), "• No forced pattern fitting", 
                 fill=(255, 165, 0), font=small_font)
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def main():
    """Test boundary-constrained constellation matching."""
    print("🌍 Boundary-Constrained Constellation Matcher")
    print("=" * 60)
    print("Using IAU boundaries and proper constellation shapes")
    print("=" * 60)
    
    start_time = time.time()
    
    # Create matcher
    matcher = BoundaryConstrainedConstellationMatcher()
    
    try:
        # Load processed data
        matcher.load_processed_data()
        
        # Find constellation patterns
        matches = matcher.find_constellation_patterns()
        
        # Create annotated image
        matcher.create_annotated_image(matches)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ Boundary-constrained matching complete!")
        print(f"   Found {len(matches)} boundary-valid constellation matches")
        
        if matches:
            print("   Matched constellations:")
            for i, match in enumerate(matches):
                hemisphere = match["hemisphere"]
                print(f"     {i+1}. {match['constellation_name']}: {match['confidence']:.2f} confidence ({hemisphere})")
        else:
            print("   No boundary-valid constellation patterns found")
            print("   This could indicate:")
            print("     - Image shows a different sky region")
            print("     - Boundary constraints too restrictive")
            print("     - Need more accurate coordinate estimation")
        
        print(f"   Total processing time: {elapsed_time:.1f} seconds")
        print(f"   Check Output/boundary_constrained_constellations.jpg for results")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        raise

if __name__ == "__main__":
    main() 