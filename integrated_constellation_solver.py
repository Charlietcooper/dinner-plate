#!/usr/bin/env python3
"""
Integrated Constellation Solver
Combines star field processing, plate solving, and ML pattern matching
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
from astroquery.astrometry_net import AstrometryNet
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntegratedConstellationSolver:
    """Integrated constellation solver with star field processing and plate solving."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('ASTROMETRY_API_KEY')
        self.star_processor = None
        self.plate_solver = None
        self.constellation_database = self._load_constellation_database()
        
    def _load_constellation_database(self) -> Dict:
        """Load constellation database."""
        try:
            with open("Processing/professional_constellation_database.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Professional database not found, using basic constellations")
            return self._create_basic_database()
    
    def _create_basic_database(self) -> Dict:
        """Create basic constellation database."""
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
                }
            }
        }
    
    def process_image(self, image_path: str) -> Dict:
        """Complete image processing pipeline."""
        logger.info("🚀 Starting integrated constellation solving...")
        
        # Step 1: Extract star field
        logger.info("Step 1: Extracting star field...")
        star_field, stars = self._extract_star_field(image_path)
        
        # Step 2: Estimate FOV
        logger.info("Step 2: Estimating field of view...")
        fov_info = self._estimate_field_of_view(stars, star_field.shape)
        
        # Step 3: Attempt plate solving
        logger.info("Step 3: Attempting plate solving...")
        plate_solve_result = self._attempt_plate_solving(image_path, fov_info)
        
        # Step 4: Match constellations
        logger.info("Step 4: Matching constellations...")
        constellation_matches = self._match_constellations(stars, plate_solve_result, fov_info)
        
        # Step 5: Create final annotation
        logger.info("Step 5: Creating final annotation...")
        self._create_final_annotation(image_path, constellation_matches, plate_solve_result, fov_info)
        
        return {
            "stars_detected": len(stars),
            "fov_estimate": fov_info.get("fov_estimate"),
            "plate_solve_success": plate_solve_result.get("success", False),
            "constellations_found": len(constellation_matches),
            "matches": constellation_matches
        }
    
    def _extract_star_field(self, image_path: str) -> Tuple[np.ndarray, List[Dict]]:
        """Extract star field using the advanced processor."""
        from advanced_star_field_processor import AdvancedStarFieldProcessor
        
        processor = AdvancedStarFieldProcessor()
        star_field, stars = processor.extract_star_field(image_path)
        
        # Save star field for later use
        cv2.imwrite("Processing/star_field_for_solving.jpg", star_field)
        
        return star_field, stars
    
    def _estimate_field_of_view(self, stars: List[Dict], image_shape: Tuple[int, int]) -> Dict:
        """Estimate field of view based on star distribution."""
        if len(stars) < 10:
            return {"fov_estimate": "unknown", "confidence": "low"}
        
        # Calculate star density
        image_area = image_shape[0] * image_shape[1]
        star_density = len(stars) / image_area
        
        # Analyze star brightness
        brightnesses = [star["brightness"] for star in stars]
        avg_brightness = np.mean(brightnesses)
        
        # Estimate FOV
        if star_density > 0.0001 and avg_brightness > 200:
            fov_estimate = "wide_field (60-90°)"
        elif star_density > 0.00005 and avg_brightness > 150:
            fov_estimate = "medium_field (20-60°)"
        elif star_density < 0.00001 and avg_brightness > 180:
            fov_estimate = "narrow_field (5-20°)"
        else:
            fov_estimate = "unknown_field"
        
        return {
            "fov_estimate": fov_estimate,
            "star_density": star_density,
            "avg_brightness": avg_brightness,
            "num_stars": len(stars),
            "confidence": "medium" if len(stars) > 50 else "low"
        }
    
    def _attempt_plate_solving(self, image_path: str, fov_info: Dict) -> Dict:
        """Attempt plate solving with improved parameters."""
        if not self.api_key:
            logger.warning("No API key available, using demo WCS")
            return self._create_demo_wcs(image_path, fov_info)
        
        try:
            logger.info("   Attempting plate solving with Astrometry.net...")
            
            # Initialize Astrometry.net
            ast = AstrometryNet()
            ast.api_key = self.api_key
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Failed to load image")
            
            # Determine scale parameters based on FOV estimate
            fov_estimate = fov_info.get("fov_estimate", "unknown")
            if "wide_field" in fov_estimate:
                scale_est = 60  # arcseconds per pixel
                scale_lower = 30
                scale_upper = 120
            elif "medium_field" in fov_estimate:
                scale_est = 20
                scale_lower = 10
                scale_upper = 40
            elif "narrow_field" in fov_estimate:
                scale_est = 5
                scale_lower = 2
                scale_upper = 10
            else:
                scale_est = 10
                scale_lower = 5
                scale_upper = 20
            
            # Calculate radius based on image size
            image_height, image_width = image.shape[:2]
            radius = min(image_width, image_height) * scale_est / 3600  # Convert to degrees
            
            logger.info(f"   Scale estimate: {scale_est} arcsec/pixel")
            logger.info(f"   Scale range: {scale_lower}-{scale_upper} arcsec/pixel")
            logger.info(f"   Search radius: {radius:.2f}°")
            
            # Attempt plate solving
            result = ast.solve_from_image(
                image_path,
                scale_est=scale_est,
                scale_lower=scale_lower,
                scale_upper=scale_upper,
                radius=radius,
                solve_timeout=120
            )
            
            if result:
                logger.info("   ✅ Plate solving successful!")
                return {
                    "success": True,
                    "wcs": result,
                    "scale_est": scale_est,
                    "radius": radius
                }
            else:
                logger.warning("   ❌ Plate solving failed, using demo WCS")
                return self._create_demo_wcs(image_path, fov_info)
                
        except Exception as e:
            logger.error(f"   ❌ Plate solving error: {e}")
            return self._create_demo_wcs(image_path, fov_info)
    
    def _create_demo_wcs(self, image_path: str, fov_info: Dict) -> Dict:
        """Create a demo WCS for testing when plate solving fails."""
        logger.info("   Creating demo WCS for testing...")
        
        # Load image to get dimensions
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Failed to load image")
        
        height, width = image.shape[:2]
        
        # Create WCS centered on southern sky
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [width//2, height//2]  # Reference pixel at center
        wcs.wcs.crval = [180, -60]  # Southern sky coordinates (RA=180°, Dec=-60°)
        
        # Set scale based on FOV estimate
        fov_estimate = fov_info.get("fov_estimate", "unknown")
        if "wide_field" in fov_estimate:
            scale = 0.1  # degrees per pixel
        elif "medium_field" in fov_estimate:
            scale = 0.05
        elif "narrow_field" in fov_estimate:
            scale = 0.01
        else:
            scale = 0.02
        
        wcs.wcs.cdelt = [scale, scale]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        
        return {
            "success": False,
            "wcs": wcs,
            "scale_est": scale * 3600,  # Convert to arcseconds
            "radius": 10,
            "demo": True
        }
    
    def _match_constellations(self, stars: List[Dict], plate_solve_result: Dict, fov_info: Dict) -> List[Dict]:
        """Match constellations using plate solving results."""
        logger.info("   Matching constellations with plate solving data...")
        
        matches = []
        wcs = plate_solve_result.get("wcs")
        
        if not wcs:
            logger.warning("   No WCS available for constellation matching")
            return matches
        
        constellations = self.constellation_database.get("constellations", {})
        
        for const_name, const_data in tqdm(constellations.items(), desc="   Matching constellations"):
            logger.info(f"   Testing {const_name}...")
            
            # Get constellation stars
            const_stars = const_data.get("stars", [])
            if len(const_stars) < 3:
                continue
            
            # Try to match this constellation
            match_result = self._match_single_constellation_with_wcs(
                const_name, const_stars, const_data, stars, wcs
            )
            
            if match_result:
                matches.append(match_result)
                logger.info(f"   ✅ Found {const_name}")
            else:
                logger.info(f"   ❌ No match for {const_name}")
        
        logger.info(f"   Total matches: {len(matches)}")
        return matches
    
    def _match_single_constellation_with_wcs(self, const_name: str, const_stars: List[Dict], 
                                            const_data: Dict, detected_stars: List[Dict], wcs: WCS) -> Optional[Dict]:
        """Match a single constellation using WCS."""
        
        # Convert constellation stars to pixel coordinates using WCS
        const_pixels = []
        for star in const_stars:
            ra, dec = star["ra"], star["dec"]
            
            # Convert RA/Dec to pixel coordinates
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
        
        if len(const_pixels) < 3:
            return None
        
        # Find matching detected stars
        matched_stars = []
        total_score = 0
        
        for const_star in const_pixels:
            best_match = None
            best_distance = float('inf')
            
            # Find closest detected star
            for detected_star in detected_stars:
                distance = np.sqrt(
                    (const_star["pixel_x"] - detected_star["x"])**2 + 
                    (const_star["pixel_y"] - detected_star["y"])**2
                )
                
                if distance < best_distance and distance < 100:  # 100 pixel tolerance
                    best_distance = distance
                    best_match = detected_star
            
            if best_match:
                matched_stars.append({
                    "constellation_star": const_star,
                    "detected_star": best_match,
                    "distance": best_distance
                })
                total_score += 1.0 / (1.0 + best_distance)
        
        if len(matched_stars) >= 3:  # Need at least 3 stars
            confidence = total_score / len(const_pixels)
            
            if confidence > 0.3:  # Minimum confidence threshold
                return {
                    "constellation": const_name,
                    "confidence": confidence,
                    "matched_stars": matched_stars,
                    "constellation_stars": const_pixels,
                    "constellation_data": const_data
                }
        
        return None
    
    def _create_final_annotation(self, image_path: str, matches: List[Dict], 
                                plate_solve_result: Dict, fov_info: Dict):
        """Create the final annotated image."""
        logger.info("   Creating final annotated image...")
        
        # Load original image
        original_image = cv2.imread(image_path)
        if original_image is None:
            logger.error("   ❌ Could not load original image")
            return
        
        # Create annotation overlay
        annotated_image = original_image.copy()
        
        # Draw matched constellations
        for match in matches:
            self._draw_constellation_annotation(annotated_image, match)
        
        # Add comprehensive information
        annotated_image = self._add_comprehensive_info(
            annotated_image, matches, plate_solve_result, fov_info
        )
        
        # Save result
        output_path = "Output/integrated_constellation_solution.jpg"
        cv2.imwrite(output_path, annotated_image)
        logger.info(f"   ✅ Saved final annotation: {output_path}")
    
    def _draw_constellation_annotation(self, image: np.ndarray, match: Dict):
        """Draw a constellation annotation."""
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
                cv2.line(image, pt1, pt2, (0, 255, 255), 3)  # Yellow line
        
        # Draw stars and labels
        for matched_star in matched_stars:
            detected_star = matched_star["detected_star"]
            const_star = matched_star["constellation_star"]
            
            # Draw star
            center = (int(detected_star["x"]), int(detected_star["y"]))
            cv2.circle(image, center, 4, (255, 255, 255), -1)  # White star
            
            # Draw label
            label = f"{const_star['name']}"
            cv2.putText(image, label, (center[0] + 8, center[1] - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw constellation name
        if matched_stars:
            center_x = int(np.mean([ms["detected_star"]["x"] for ms in matched_stars]))
            center_y = int(np.mean([ms["detected_star"]["y"] for ms in matched_stars]))
            
            label = f"{const_name} ({confidence:.2f})"
            cv2.putText(image, label, (center_x - 30, center_y - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    def _add_comprehensive_info(self, image: np.ndarray, matches: List[Dict], 
                               plate_solve_result: Dict, fov_info: Dict) -> np.ndarray:
        """Add comprehensive information to the image."""
        # Convert to PIL for better text rendering
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 28)
            small_font = ImageFont.truetype("arial.ttf", 18)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Add title
        title = "Integrated Constellation Recognition System"
        draw.text((20, 20), title, fill=(255, 255, 255), font=font)
        
        # Add processing information
        y_offset = 70
        info_lines = [
            f"Stars detected: {fov_info.get('num_stars', 0)}",
            f"FOV estimate: {fov_info.get('fov_estimate', 'unknown')}",
            f"Plate solve: {'✅ Success' if plate_solve_result.get('success') else '❌ Demo WCS'}",
            f"Constellations found: {len(matches)}"
        ]
        
        for line in info_lines:
            draw.text((20, y_offset), line, fill=(255, 255, 255), font=small_font)
            y_offset += 25
        
        # Add constellation details
        y_offset += 10
        for match in matches:
            const_name = match["constellation"]
            confidence = match["confidence"]
            num_stars = len(match["matched_stars"])
            
            line = f"• {const_name}: {confidence:.2f} ({num_stars} stars)"
            draw.text((20, y_offset), line, fill=(0, 255, 255), font=small_font)
            y_offset += 22
        
        # Add technical details
        y_offset += 20
        if plate_solve_result.get("demo"):
            draw.text((20, y_offset), "Note: Using demo WCS (plate solving failed)", 
                     fill=(255, 165, 0), font=small_font)  # Orange
        else:
            scale_est = plate_solve_result.get("scale_est", 0)
            draw.text((20, y_offset), f"Scale: {scale_est:.1f} arcsec/pixel", 
                     fill=(255, 255, 255), font=small_font)
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def main():
    """Test the integrated constellation solver."""
    print("🚀 Integrated Constellation Solver Test")
    print("=" * 60)
    print("Testing complete pipeline: star extraction + plate solving + ML matching")
    print("=" * 60)
    
    start_time = time.time()
    
    # Create solver
    solver = IntegratedConstellationSolver()
    
    # Test image path
    test_image_path = "Input/test-1.jpg"
    
    try:
        # Process image
        result = solver.process_image(test_image_path)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ Integrated constellation solving complete!")
        print(f"   Stars detected: {result['stars_detected']}")
        print(f"   FOV estimate: {result['fov_estimate']}")
        print(f"   Plate solve success: {result['plate_solve_success']}")
        print(f"   Constellations found: {result['constellations_found']}")
        
        if result['matches']:
            print("   Matched constellations:")
            for match in result['matches']:
                print(f"     • {match['constellation']}: {match['confidence']:.2f} confidence")
        else:
            print("   No constellations matched (this is expected with demo WCS)")
        
        print(f"   Total processing time: {elapsed_time:.1f} seconds")
        print(f"   Check Output/integrated_constellation_solution.jpg for results")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        raise

if __name__ == "__main__":
    main() 