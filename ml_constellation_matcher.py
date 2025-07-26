#!/usr/bin/env python3
"""
Machine Learning Constellation Pattern Matcher
Uses processed star field data to identify constellations with proper scale and FOV
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import json
import time
from typing import List, Tuple, Dict, Optional
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLConstellationMatcher:
    """Machine learning constellation pattern matcher with scale and FOV handling."""
    
    def __init__(self):
        self.constellation_database = self._load_constellation_database()
        self.ml_dataset = None
        self.detected_stars = None
        self.fov_info = None
        
    def _load_constellation_database(self) -> Dict:
        """Load the professional constellation database."""
        try:
            with open("Processing/professional_constellation_database.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Professional database not found, using basic constellations")
            return self._create_basic_database()
    
    def _create_basic_database(self) -> Dict:
        """Create a basic constellation database for testing."""
        return {
            "constellations": {
                "Crux": {
                    "name": "Crux",
                    "hemisphere": "southern",
                    "stars": [
                        {"name": "Acrux", "ra": 186.6495, "dec": -63.0991, "mag": 0.77},
                        {"name": "Mimosa", "ra": 191.9303, "dec": -59.6888, "mag": 1.25},
                        {"name": "Gacrux", "ra": 187.7915, "dec": -57.1133, "mag": 1.59},
                        {"name": "Delta Crucis", "ra": 183.7863, "dec": -58.7489, "mag": 2.79}
                    ],
                    "lines": [["Acrux", "Mimosa"], ["Mimosa", "Gacrux"], ["Gacrux", "Delta Crucis"], ["Delta Crucis", "Acrux"]]
                },
                "Carina": {
                    "name": "Carina", 
                    "hemisphere": "southern",
                    "stars": [
                        {"name": "Canopus", "ra": 95.9880, "dec": -52.6957, "mag": -0.74},
                        {"name": "Avior", "ra": 139.2725, "dec": -59.5092, "mag": 1.86},
                        {"name": "Aspidiske", "ra": 139.2725, "dec": -59.5092, "mag": 2.21}
                    ],
                    "lines": [["Canopus", "Avior"], ["Avior", "Aspidiske"]]
                }
            }
        }
    
    def load_processed_data(self, dataset_path: str = "Processing/ml_dataset.json"):
        """Load the processed star field data."""
        logger.info("📊 Loading processed star field data...")
        
        try:
            with open(dataset_path, 'r') as f:
                self.ml_dataset = json.load(f)
            
            # Load additional data
            with open("Processing/detected_stars.json", 'r') as f:
                self.detected_stars = json.load(f)
            
            with open("Processing/fov_estimation.json", 'r') as f:
                self.fov_info = json.load(f)
            
            logger.info(f"   Loaded {len(self.detected_stars)} stars")
            logger.info(f"   FOV estimate: {self.fov_info.get('fov_estimate', 'unknown')}")
            logger.info(f"   Pattern candidates: {len(self.ml_dataset.get('pattern_candidates', []))}")
            
        except FileNotFoundError as e:
            logger.error(f"❌ Failed to load processed data: {e}")
            raise
    
    def estimate_image_scale_and_orientation(self) -> Dict:
        """Estimate image scale and orientation based on star patterns."""
        logger.info("🔍 Estimating image scale and orientation...")
        
        if not self.detected_stars:
            raise ValueError("No detected stars available")
        
        # Calculate star distribution statistics
        x_coords = [star["x"] for star in self.detected_stars]
        y_coords = [star["y"] for star in self.detected_stars]
        
        # Calculate image center and dimensions
        image_center_x = np.mean(x_coords)
        image_center_y = np.mean(y_coords)
        image_width = max(x_coords) - min(x_coords)
        image_height = max(y_coords) - min(y_coords)
        
        # Estimate pixel scale based on FOV
        fov_estimate = self.fov_info.get("fov_estimate", "unknown")
        if "narrow_field" in fov_estimate:
            estimated_fov_degrees = 10  # 5-20° range
        elif "medium_field" in fov_estimate:
            estimated_fov_degrees = 40  # 20-60° range
        elif "wide_field" in fov_estimate:
            estimated_fov_degrees = 75  # 60-90° range
        else:
            estimated_fov_degrees = 20  # Default
        
        # Calculate scale (degrees per pixel)
        scale_degrees_per_pixel = estimated_fov_degrees / max(image_width, image_height)
        
        # Estimate image center in sky coordinates (rough approximation)
        # This would normally come from plate solving
        estimated_ra_center = 180.0  # Southern sky
        estimated_dec_center = -60.0
        
        scale_info = {
            "image_center_pixel": (image_center_x, image_center_y),
            "image_dimensions": (image_width, image_height),
            "estimated_fov_degrees": estimated_fov_degrees,
            "scale_degrees_per_pixel": scale_degrees_per_pixel,
            "estimated_sky_center": (estimated_ra_center, estimated_dec_center),
            "confidence": "medium"
        }
        
        logger.info(f"   Estimated FOV: {estimated_fov_degrees}°")
        logger.info(f"   Scale: {scale_degrees_per_pixel:.6f}°/pixel")
        logger.info(f"   Image center: ({image_center_x:.1f}, {image_center_y:.1f})")
        
        return scale_info
    
    def match_constellation_patterns(self, scale_info: Dict) -> List[Dict]:
        """Match detected star patterns to known constellations."""
        logger.info("🎯 Matching constellation patterns...")
        
        matches = []
        constellations = self.constellation_database.get("constellations", {})
        
        for const_name, const_data in tqdm(constellations.items(), desc="   Matching constellations"):
            logger.info(f"   Testing constellation: {const_name}")
            
            # Get constellation stars
            const_stars = const_data.get("stars", [])
            if len(const_stars) < 3:
                continue
            
            # Try to match this constellation
            match_result = self._match_single_constellation(
                const_name, const_stars, const_data, scale_info
            )
            
            if match_result:
                matches.append(match_result)
                logger.info(f"   ✅ Found match for {const_name}")
            else:
                logger.info(f"   ❌ No match for {const_name}")
        
        logger.info(f"   Total matches found: {len(matches)}")
        return matches
    
    def _match_single_constellation(self, const_name: str, const_stars: List[Dict], 
                                   const_data: Dict, scale_info: Dict) -> Optional[Dict]:
        """Match a single constellation to detected stars."""
        
        # Convert constellation stars to pixel coordinates
        const_pixels = []
        for star in const_stars:
            ra, dec = star["ra"], star["dec"]
            pixel_pos = self._sky_to_pixel(ra, dec, scale_info)
            if pixel_pos:
                const_pixels.append({
                    "name": star["name"],
                    "ra": ra,
                    "dec": dec,
                    "pixel_x": pixel_pos[0],
                    "pixel_y": pixel_pos[1],
                    "magnitude": star.get("mag", 0)
                })
        
        if len(const_pixels) < 3:
            return None
        
        # Find best matching stars in detected stars
        best_match = self._find_best_star_match(const_pixels, const_data)
        
        if best_match:
            return {
                "constellation": const_name,
                "confidence": best_match["confidence"],
                "matched_stars": best_match["matched_stars"],
                "constellation_stars": const_pixels,
                "transformation": best_match["transformation"]
            }
        
        return None
    
    def _sky_to_pixel(self, ra: float, dec: float, scale_info: Dict) -> Optional[Tuple[float, float]]:
        """Convert sky coordinates to pixel coordinates."""
        try:
            # Get scale and center information
            scale = scale_info["scale_degrees_per_pixel"]
            center_ra, center_dec = scale_info["estimated_sky_center"]
            center_x, center_y = scale_info["image_center_pixel"]
            
            # Calculate offset from center
            ra_offset = (ra - center_ra) * np.cos(np.radians(center_dec))
            dec_offset = dec - center_dec
            
            # Convert to pixels
            pixel_x = center_x + (ra_offset / scale)
            pixel_y = center_y - (dec_offset / scale)  # Y-axis is inverted
            
            return (pixel_x, pixel_y)
        except Exception as e:
            logger.debug(f"Failed to convert coordinates: {e}")
            return None
    
    def _find_best_star_match(self, const_pixels: List[Dict], const_data: Dict) -> Optional[Dict]:
        """Find the best matching stars for a constellation."""
        
        # Get constellation lines for shape matching
        const_lines = const_data.get("lines", [])
        
        # Create constellation shape signature
        const_shape = self._create_shape_signature(const_pixels, const_lines)
        
        # Try different transformations and find best match
        best_match = None
        best_score = 0
        
        # Sample different positions and scales
        for scale_factor in [0.5, 1.0, 1.5, 2.0]:
            for dx in range(-100, 101, 50):  # Translation in X
                for dy in range(-100, 101, 50):  # Translation in Y
                    
                    # Apply transformation
                    transformed_pixels = []
                    for star in const_pixels:
                        transformed_pixels.append({
                            "x": star["pixel_x"] * scale_factor + dx,
                            "y": star["pixel_y"] * scale_factor + dy,
                            "name": star["name"]
                        })
                    
                    # Find matching detected stars
                    match_result = self._find_matching_stars(transformed_pixels)
                    
                    if match_result and match_result["score"] > best_score:
                        best_score = match_result["score"]
                        best_match = {
                            "confidence": match_result["score"],
                            "matched_stars": match_result["matched_stars"],
                            "transformation": {
                                "scale": scale_factor,
                                "translation": (dx, dy)
                            }
                        }
        
        return best_match if best_score > 0.3 else None
    
    def _create_shape_signature(self, stars: List[Dict], lines: List[List[str]]) -> Dict:
        """Create a shape signature for pattern matching."""
        signature = {
            "star_positions": [(s["pixel_x"], s["pixel_y"]) for s in stars],
            "line_lengths": [],
            "angles": []
        }
        
        # Calculate line lengths
        for line in lines:
            star1_name, star2_name = line
            star1 = next((s for s in stars if s["name"] == star1_name), None)
            star2 = next((s for s in stars if s["name"] == star2_name), None)
            
            if star1 and star2:
                dx = star2["pixel_x"] - star1["pixel_x"]
                dy = star2["pixel_y"] - star1["pixel_y"]
                length = np.sqrt(dx*dx + dy*dy)
                signature["line_lengths"].append(length)
        
        return signature
    
    def _find_matching_stars(self, transformed_pixels: List[Dict]) -> Optional[Dict]:
        """Find detected stars that match transformed constellation stars."""
        if not self.detected_stars:
            return None
        
        matched_stars = []
        total_score = 0
        
        for const_star in transformed_pixels:
            best_match = None
            best_distance = float('inf')
            
            # Find closest detected star
            for detected_star in self.detected_stars:
                distance = np.sqrt(
                    (const_star["x"] - detected_star["x"])**2 + 
                    (const_star["y"] - detected_star["y"])**2
                )
                
                if distance < best_distance and distance < 50:  # 50 pixel tolerance
                    best_distance = distance
                    best_match = detected_star
            
            if best_match:
                matched_stars.append({
                    "constellation_star": const_star,
                    "detected_star": best_match,
                    "distance": best_distance
                })
                total_score += 1.0 / (1.0 + best_distance)  # Score based on distance
        
        if len(matched_stars) >= 3:  # Need at least 3 stars for a match
            return {
                "score": total_score / len(transformed_pixels),
                "matched_stars": matched_stars
            }
        
        return None
    
    def create_annotation_image(self, matches: List[Dict], output_path: str = "Output/ml_constellation_annotation.jpg"):
        """Create an annotated image showing matched constellations."""
        logger.info("🎨 Creating annotated image...")
        
        # Load original image
        original_image = cv2.imread("Input/test-1.jpg")
        if original_image is None:
            logger.error("❌ Could not load original image")
            return
        
        # Create annotation overlay
        annotated_image = original_image.copy()
        
        # Draw matched constellations
        for match in matches:
            self._draw_constellation_match(annotated_image, match)
        
        # Add title and information
        annotated_image = self._add_annotation_info(annotated_image, matches)
        
        # Save result
        cv2.imwrite(output_path, annotated_image)
        logger.info(f"   Saved annotated image: {output_path}")
    
    def _draw_constellation_match(self, image: np.ndarray, match: Dict):
        """Draw a matched constellation on the image."""
        const_name = match["constellation"]
        matched_stars = match["matched_stars"]
        confidence = match["confidence"]
        
        # Draw constellation lines
        const_stars = match["constellation_stars"]
        const_data = self.constellation_database["constellations"][const_name]
        lines = const_data.get("lines", [])
        
        for line in lines:
            star1_name, star2_name = line
            star1 = next((s for s in const_stars if s["name"] == star1_name), None)
            star2 = next((s for s in const_stars if s["name"] == star2_name), None)
            
            if star1 and star2:
                # Find corresponding detected stars
                detected1 = next((ms["detected_star"] for ms in matched_stars 
                                if ms["constellation_star"]["name"] == star1_name), None)
                detected2 = next((ms["detected_star"] for ms in matched_stars 
                                if ms["constellation_star"]["name"] == star2_name), None)
                
                if detected1 and detected2:
                    # Draw line between detected stars
                    pt1 = (int(detected1["x"]), int(detected1["y"]))
                    pt2 = (int(detected2["x"]), int(detected2["y"]))
                    cv2.line(image, pt1, pt2, (0, 255, 255), 2)  # Yellow line
        
        # Draw stars and labels
        for matched_star in matched_stars:
            detected_star = matched_star["detected_star"]
            const_star = matched_star["constellation_star"]
            
            # Draw star
            center = (int(detected_star["x"]), int(detected_star["y"]))
            cv2.circle(image, center, 3, (255, 255, 255), -1)  # White star
            
            # Draw label
            label = f"{const_star['name']}"
            cv2.putText(image, label, (center[0] + 5, center[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw constellation name
        if matched_stars:
            center_x = int(np.mean([ms["detected_star"]["x"] for ms in matched_stars]))
            center_y = int(np.mean([ms["detected_star"]["y"] for ms in matched_stars]))
            
            label = f"{const_name} ({confidence:.2f})"
            cv2.putText(image, label, (center_x - 20, center_y - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    def _add_annotation_info(self, image: np.ndarray, matches: List[Dict]) -> np.ndarray:
        """Add title and information to the annotated image."""
        # Convert to PIL for better text rendering
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 24)
            small_font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Add title
        title = "Machine Learning Constellation Recognition"
        draw.text((20, 20), title, fill=(255, 255, 255), font=font)
        
        # Add match information
        y_offset = 60
        draw.text((20, y_offset), f"Constellations found: {len(matches)}", 
                 fill=(255, 255, 255), font=small_font)
        y_offset += 25
        
        for match in matches:
            const_name = match["constellation"]
            confidence = match["confidence"]
            draw.text((20, y_offset), f"• {const_name}: {confidence:.2f}", 
                     fill=(0, 255, 255), font=small_font)
            y_offset += 20
        
        # Add FOV info
        if self.fov_info:
            fov_estimate = self.fov_info.get("fov_estimate", "unknown")
            draw.text((20, y_offset + 10), f"FOV Estimate: {fov_estimate}", 
                     fill=(255, 255, 255), font=small_font)
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def main():
    """Test machine learning constellation matching."""
    print("🤖 Machine Learning Constellation Matcher Test")
    print("=" * 60)
    print("Testing ML-based constellation pattern recognition")
    print("=" * 60)
    
    start_time = time.time()
    
    # Create matcher
    matcher = MLConstellationMatcher()
    
    try:
        # Load processed data
        matcher.load_processed_data()
        
        # Estimate scale and orientation
        scale_info = matcher.estimate_image_scale_and_orientation()
        
        # Match constellations
        matches = matcher.match_constellation_patterns(scale_info)
        
        # Create annotated image
        matcher.create_annotation_image(matches)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ Machine learning constellation matching complete!")
        print(f"   Found {len(matches)} constellation matches")
        for match in matches:
            print(f"   • {match['constellation']}: {match['confidence']:.2f} confidence")
        print(f"   Total processing time: {elapsed_time:.1f} seconds")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        raise

if __name__ == "__main__":
    main() 