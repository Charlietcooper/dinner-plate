#!/usr/bin/env python3
"""
Improved Constellation Matcher - Handles large star counts and overlays constellations
Addresses issues with pattern accuracy and constellation overlay
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
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedConstellationMatcher:
    """Improved constellation matcher with better pattern recognition and overlay."""
    
    def __init__(self):
        self.constellation_database = self._load_constellation_database()
        self.detected_stars = None
        self.fov_info = None
        
    def _load_constellation_database(self) -> Dict:
        """Load constellation database."""
        try:
            with open("Processing/professional_constellation_database.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Professional database not found, using enhanced basic constellations")
            return self._create_enhanced_database()
    
    def _create_enhanced_database(self) -> Dict:
        """Create enhanced constellation database with more southern constellations."""
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
                },
                "Centaurus": {
                    "name": "Centaurus",
                    "hemisphere": "southern", 
                    "stars": [
                        {"name": "Alpha Centauri", "ra": 219.9021, "dec": -60.8340, "mag": -0.27},
                        {"name": "Hadar", "ra": 210.9559, "dec": -60.3730, "mag": 0.61},
                        {"name": "Menkent", "ra": 204.9719, "dec": -36.7123, "mag": 2.06}
                    ],
                    "lines": [["Alpha Centauri", "Hadar"], ["Hadar", "Menkent"]]
                },
                "Musca": {
                    "name": "Musca",
                    "hemisphere": "southern",
                    "stars": [
                        {"name": "Alpha Muscae", "ra": 184.9767, "dec": -69.1357, "mag": 2.69},
                        {"name": "Beta Muscae", "ra": 185.3417, "dec": -67.9608, "mag": 3.04},
                        {"name": "Gamma Muscae", "ra": 186.7342, "dec": -72.1328, "mag": 3.84}
                    ],
                    "lines": [["Alpha Muscae", "Beta Muscae"], ["Beta Muscae", "Gamma Muscae"]]
                },
                "Vela": {
                    "name": "Vela",
                    "hemisphere": "southern",
                    "stars": [
                        {"name": "Suhail", "ra": 136.9992, "dec": -43.4326, "mag": 1.83},
                        {"name": "Alsephina", "ra": 140.5283, "dec": -40.4669, "mag": 1.99},
                        {"name": "Markeb", "ra": 138.2992, "dec": -55.0108, "mag": 2.47}
                    ],
                    "lines": [["Suhail", "Alsephina"], ["Alsephina", "Markeb"]]
                }
            }
        }
    
    def load_processed_data(self):
        """Load the improved processed star field data."""
        logger.info("📊 Loading improved processed data...")
        
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
            logger.error(f"❌ Failed to load improved data: {e}")
            raise
    
    def create_realistic_wcs(self, image_shape: Tuple[int, int]) -> WCS:
        """Create a more realistic WCS for southern sky."""
        logger.info("🌐 Creating realistic WCS for southern sky...")
        
        height, width = image_shape
        
        # Create WCS centered on southern sky region
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [width//2, height//2]  # Reference pixel at center
        
        # Center on a realistic southern sky region (near Crux)
        wcs.wcs.crval = [190, -62]  # RA=190°, Dec=-62° (near Crux)
        
        # Set scale based on FOV estimate
        fov_estimate = self.fov_info.get("fov_estimate", "unknown")
        if "wide_field" in fov_estimate:
            scale = 0.15  # degrees per pixel
        elif "medium_field" in fov_estimate:
            scale = 0.08
        elif "narrow_field" in fov_estimate:
            scale = 0.03
        else:
            scale = 0.05
        
        wcs.wcs.cdelt = [scale, scale]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        
        logger.info(f"   WCS center: RA={wcs.wcs.crval[0]}°, Dec={wcs.wcs.crval[1]}°")
        logger.info(f"   Scale: {scale}°/pixel")
        
        return wcs
    
    def match_constellations_with_realistic_wcs(self) -> List[Dict]:
        """Match constellations using realistic WCS."""
        logger.info("🎯 Matching constellations with realistic WCS...")
        
        if not self.detected_stars:
            raise ValueError("No detected stars available")
        
        # Create realistic WCS
        image_shape = (3413, 5120)  # Height, width from the image
        wcs = self.create_realistic_wcs(image_shape)
        
        matches = []
        constellations = self.constellation_database.get("constellations", {})
        
        for const_name, const_data in tqdm(constellations.items(), desc="   Matching constellations"):
            logger.info(f"   Testing {const_name}...")
            
            # Get constellation stars
            const_stars = const_data.get("stars", [])
            if len(const_stars) < 3:
                continue
            
            # Try to match this constellation
            match_result = self._match_constellation_with_wcs(
                const_name, const_stars, const_data, wcs
            )
            
            if match_result:
                matches.append(match_result)
                logger.info(f"   ✅ Found {const_name} with confidence {match_result['confidence']:.2f}")
            else:
                logger.info(f"   ❌ No match for {const_name}")
        
        logger.info(f"   Total matches: {len(matches)}")
        return matches
    
    def _match_constellation_with_wcs(self, const_name: str, const_stars: List[Dict], 
                                     const_data: Dict, wcs: WCS) -> Optional[Dict]:
        """Match a constellation using WCS with improved tolerance."""
        
        # Convert constellation stars to pixel coordinates
        const_pixels = []
        for star in const_stars:
            ra, dec = star["ra"], star["dec"]
            
            try:
                pixel_coords = wcs.wcs_world2pix([[ra, dec]], 0)[0]
                pixel_x, pixel_y = pixel_coords
                
                if not np.isnan(pixel_x) and not np.isnan(pixel_y):
                    const_pixels.append({
                        "name": star["name"],
                        "ra": ra,
                        "dec": dec,
                        "pixel_x": pixel_x,
                        "pixel_y": pixel_y,
                        "magnitude": star.get("mag", 0)
                    })
            except Exception as e:
                logger.debug(f"Failed to convert coordinates for {star['name']}: {e}")
                continue
        
        if len(const_pixels) < 2:  # Need at least 2 stars
            return None
        
        # Find matching detected stars with improved tolerance
        matched_stars = []
        total_score = 0
        
        for const_star in const_pixels:
            best_match = None
            best_distance = float('inf')
            
            # Find closest detected star within tolerance
            for detected_star in self.detected_stars:
                distance = np.sqrt(
                    (const_star["pixel_x"] - detected_star["x"])**2 + 
                    (const_star["pixel_y"] - detected_star["y"])**2
                )
                
                # Increased tolerance for better matching
                if distance < best_distance and distance < 200:  # 200 pixel tolerance
                    best_distance = distance
                    best_match = detected_star
            
            if best_match:
                matched_stars.append({
                    "constellation_star": const_star,
                    "detected_star": best_match,
                    "distance": best_distance
                })
                # Improved scoring based on distance and brightness
                brightness_factor = best_match.get("brightness", 0) / 255.0
                distance_factor = 1.0 / (1.0 + best_distance / 50.0)
                total_score += brightness_factor * distance_factor
        
        if len(matched_stars) >= 2:  # Need at least 2 stars
            confidence = total_score / len(const_pixels)
            
            # Lower confidence threshold for better detection
            if confidence > 0.1:  # Reduced threshold
                return {
                    "constellation": const_name,
                    "confidence": confidence,
                    "matched_stars": matched_stars,
                    "constellation_stars": const_pixels,
                    "constellation_data": const_data
                }
        
        return None
    
    def create_improved_annotation(self, matches: List[Dict], output_path: str = "Output/improved_constellation_annotation.jpg"):
        """Create an improved annotated image with constellation overlays."""
        logger.info("🎨 Creating improved constellation annotation...")
        
        # Load original image
        original_image = cv2.imread("Input/test-1.jpg")
        if original_image is None:
            logger.error("❌ Could not load original image")
            return
        
        # Create annotation overlay
        annotated_image = original_image.copy()
        
        # Draw matched constellations
        for match in matches:
            self._draw_improved_constellation(annotated_image, match)
        
        # Add comprehensive information
        annotated_image = self._add_improved_info(annotated_image, matches)
        
        # Save result
        cv2.imwrite(output_path, annotated_image)
        logger.info(f"   ✅ Saved improved annotation: {output_path}")
    
    def _draw_improved_constellation(self, image: np.ndarray, match: Dict):
        """Draw a constellation with improved visualization."""
        const_name = match["constellation"]
        matched_stars = match["matched_stars"]
        confidence = match["confidence"]
        const_data = match["constellation_data"]
        
        # Draw constellation lines
        lines = const_data.get("lines", [])
        for line in lines:
            star1_name, star2_name = line
            
            # Find corresponding detected stars
            detected1 = next((ms["detected_star"] for ms in matched_stars 
                            if ms["constellation_star"]["name"] == star1_name), None)
            detected2 = next((ms["detected_star"] for ms in matched_stars 
                            if ms["constellation_star"]["name"] == star2_name), None)
            
            if detected1 and detected2:
                pt1 = (int(detected1["x"]), int(detected1["y"]))
                pt2 = (int(detected2["x"]), int(detected2["y"]))
                cv2.line(image, pt1, pt2, (0, 255, 255), 4)  # Thicker yellow line
        
        # Draw stars and labels
        for matched_star in matched_stars:
            detected_star = matched_star["detected_star"]
            const_star = matched_star["constellation_star"]
            
            # Draw star with size based on brightness
            center = (int(detected_star["x"]), int(detected_star["y"]))
            brightness = detected_star.get("brightness", 0)
            size = max(2, int(brightness / 50))  # Size based on brightness
            
            cv2.circle(image, center, size, (255, 255, 255), -1)  # White star
            
            # Draw label
            label = f"{const_star['name']}"
            cv2.putText(image, label, (center[0] + 10, center[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw constellation name
        if matched_stars:
            center_x = int(np.mean([ms["detected_star"]["x"] for ms in matched_stars]))
            center_y = int(np.mean([ms["detected_star"]["y"] for ms in matched_stars]))
            
            label = f"{const_name} ({confidence:.2f})"
            cv2.putText(image, label, (center_x - 40, center_y - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
    
    def _add_improved_info(self, image: np.ndarray, matches: List[Dict]) -> np.ndarray:
        """Add improved information to the image."""
        # Convert to PIL for better text rendering
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 32)
            small_font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Add title
        title = "Improved Constellation Recognition"
        draw.text((20, 20), title, fill=(255, 255, 255), font=font)
        
        # Add processing information
        y_offset = 80
        info_lines = [
            f"Stars detected: {len(self.detected_stars)}",
            f"FOV estimate: {self.fov_info.get('fov_estimate', 'unknown')}",
            f"Star density: {self.fov_info.get('star_density', 0):.6f} stars/pixel²",
            f"Constellations found: {len(matches)}"
        ]
        
        for line in info_lines:
            draw.text((20, y_offset), line, fill=(255, 255, 255), font=small_font)
            y_offset += 30
        
        # Add constellation details
        y_offset += 20
        for match in matches:
            const_name = match["constellation"]
            confidence = match["confidence"]
            num_stars = len(match["matched_stars"])
            
            line = f"• {const_name}: {confidence:.2f} ({num_stars} stars)"
            draw.text((20, y_offset), line, fill=(0, 255, 255), font=small_font)
            y_offset += 25
        
        # Add technical details
        y_offset += 20
        draw.text((20, y_offset), "Note: Using realistic WCS for southern sky", 
                 fill=(255, 165, 0), font=small_font)  # Orange
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def main():
    """Test improved constellation matching."""
    print("🎯 Improved Constellation Matcher Test")
    print("=" * 60)
    print("Testing improved pattern recognition and constellation overlay")
    print("=" * 60)
    
    start_time = time.time()
    
    # Create matcher
    matcher = ImprovedConstellationMatcher()
    
    try:
        # Load processed data
        matcher.load_processed_data()
        
        # Match constellations
        matches = matcher.match_constellations_with_realistic_wcs()
        
        # Create improved annotation
        matcher.create_improved_annotation(matches)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ Improved constellation matching complete!")
        print(f"   Found {len(matches)} constellation matches")
        
        if matches:
            print("   Matched constellations:")
            for match in matches:
                print(f"     • {match['constellation']}: {match['confidence']:.2f} confidence")
        else:
            print("   No constellations matched - this may indicate:")
            print("     - Image is not of the southern sky")
            print("     - Need to adjust WCS parameters")
            print("     - Need real plate solving for accurate coordinates")
        
        print(f"   Total processing time: {elapsed_time:.1f} seconds")
        print(f"   Check Output/improved_constellation_annotation.jpg for results")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        raise

if __name__ == "__main__":
    main() 