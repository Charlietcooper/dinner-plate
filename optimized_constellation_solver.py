#!/usr/bin/env python3
"""
Optimized Constellation Solver - Reduced sensitivity, magnitude limits, better nebulosity handling
Addresses issues with too many stars, no constellation matches, and black nebulosity areas
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

class OptimizedConstellationSolver:
    """Optimized constellation solver with reduced sensitivity and better nebulosity handling."""
    
    def __init__(self):
        self.constellation_database = self._load_constellation_database()
        self.detected_stars = None
        self.fov_info = None
        
        # **OPTIMIZED PARAMETERS** - Reduced sensitivity for constellation recognition
        self.min_magnitude = 150  # Higher brightness threshold (was 100)
        self.max_stars_for_matching = 500  # Limit stars for constellation matching
        self.star_size_min = 3  # Larger minimum star size (was 1)
        self.star_size_max = 30  # Smaller maximum star size (was 50)
        self.nebulosity_preservation = 0.2  # Preserve some nebulosity (was 0.4 removal)
        
    def _load_constellation_database(self) -> Dict:
        """Load constellation database."""
        try:
            with open("Processing/professional_constellation_database.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Professional database not found, using optimized constellations")
            return self._create_optimized_database()
    
    def _create_optimized_database(self) -> Dict:
        """Create optimized constellation database with bright stars only."""
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
    
    def extract_optimized_star_field(self, image_path: str) -> Tuple[np.ndarray, List[Dict]]:
        """Extract star field with optimized parameters for constellation recognition."""
        logger.info("🌟 Optimized star field extraction...")
        
        # Load image
        logger.info(f"   Loading image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        logger.info(f"   Original image size: {image.shape[1]}x{image.shape[0]}")
        
        # Convert to grayscale
        logger.info("   Converting to grayscale...")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # **OPTIMIZED** nebulosity handling - preserve some nebulosity
        logger.info("   Optimized nebulosity handling...")
        star_field = self._optimized_nebulosity_handling(gray)
        
        # **OPTIMIZED** star detection with higher thresholds
        logger.info("   Optimized star detection...")
        stars = self._optimized_star_detection(star_field)
        
        # **OPTIMIZED** filter stars for constellation recognition
        logger.info("   Filtering stars for constellation recognition...")
        filtered_stars = self._filter_stars_for_constellations(stars)
        
        # Create optimized star field image
        optimized_star_field = self._create_optimized_star_field(star_field, filtered_stars)
        
        logger.info(f"   Detected {len(stars)} total stars")
        logger.info(f"   Filtered to {len(filtered_stars)} stars for constellation recognition")
        
        return optimized_star_field, filtered_stars
    
    def _optimized_nebulosity_handling(self, gray_image: np.ndarray) -> np.ndarray:
        """Optimized nebulosity handling that preserves some nebulosity."""
        # **REDUCED** nebulosity removal to preserve extended objects
        kernel_size = int(min(gray_image.shape) * 0.05)  # Smaller kernel (was 0.08)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        logger.info(f"   Using reduced kernel size: {kernel_size}x{kernel_size}")
        
        # Apply Gaussian blur to get nebulosity
        nebulosity = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)
        
        # **REDUCED** nebulosity subtraction to preserve some extended objects
        star_field = cv2.addWeighted(gray_image, 1.0, nebulosity, -self.nebulosity_preservation, 0)
        
        # **ADDED** contrast enhancement to bring out stars while preserving nebulosity
        star_field = cv2.equalizeHist(star_field)
        
        # **ADDED** mild noise reduction
        star_field = cv2.medianBlur(star_field, 3)
        
        # Normalize
        star_field = np.clip(star_field, 0, 255).astype(np.uint8)
        
        return star_field
    
    def _optimized_star_detection(self, star_field: np.ndarray) -> List[Dict]:
        """Optimized star detection with higher thresholds."""
        stars = []
        
        # **OPTIMIZED** Method 1: Adaptive thresholding with higher threshold
        logger.info("   Method 1: Optimized adaptive thresholding...")
        adaptive_thresh = cv2.adaptiveThreshold(
            star_field, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5  # Higher parameters
        )
        contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        logger.info(f"   Found {len(contours)} contours with optimized adaptive threshold")
        
        for contour in tqdm(contours, desc="   Processing optimized contours"):
            area = cv2.contourArea(contour)
            if self.star_size_min < area < self.star_size_max:  # **OPTIMIZED** size range
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Calculate brightness
                    y1, y2 = max(0, cy-3), min(star_field.shape[0], cy+4)
                    x1, x2 = max(0, cx-3), min(star_field.shape[1], cx+4)
                    region = star_field[y1:y2, x1:x2]
                    brightness = np.mean(region) if region.size > 0 else 0
                    
                    if brightness > self.min_magnitude:  # **OPTIMIZED** higher threshold
                        stars.append({
                            "x": cx,
                            "y": cy,
                            "brightness": brightness,
                            "area": area,
                            "method": "optimized_adaptive"
                        })
        
        logger.info(f"   Optimized adaptive threshold found {len([s for s in stars if s['method'] == 'optimized_adaptive'])} stars")
        
        # **OPTIMIZED** Method 2: Simple thresholding with higher threshold
        logger.info("   Method 2: Optimized simple thresholding...")
        _, thresh = cv2.threshold(star_field, int(255 * 0.7), 255, cv2.THRESH_BINARY)  # **HIGHER** threshold (was 0.6)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        logger.info(f"   Found {len(contours)} contours with optimized simple threshold")
        
        for contour in tqdm(contours, desc="   Processing optimized simple contours"):
            area = cv2.contourArea(contour)
            if self.star_size_min < area < self.star_size_max:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Calculate brightness
                    y1, y2 = max(0, cy-3), min(star_field.shape[0], cy+4)
                    x1, x2 = max(0, cx-3), min(star_field.shape[1], cx+4)
                    region = star_field[y1:y2, x1:x2]
                    brightness = np.mean(region) if region.size > 0 else 0
                    
                    if brightness > self.min_magnitude:
                        stars.append({
                            "x": cx,
                            "y": cy,
                            "brightness": brightness,
                            "area": area,
                            "method": "optimized_simple"
                        })
        
        logger.info(f"   Optimized simple threshold found {len([s for s in stars if s['method'] == 'optimized_simple'])} stars")
        
        # Remove duplicates
        unique_stars = self._remove_duplicate_stars(stars)
        
        logger.info(f"   Final unique stars: {len(unique_stars)}")
        
        return unique_stars
    
    def _remove_duplicate_stars(self, stars: List[Dict]) -> List[Dict]:
        """Remove duplicate star detections."""
        unique_stars = []
        seen_positions = set()
        
        for star in stars:
            pos = (star["x"], star["y"])
            if pos not in seen_positions:
                seen_positions.add(pos)
                unique_stars.append(star)
        
        return unique_stars
    
    def _filter_stars_for_constellations(self, stars: List[Dict]) -> List[Dict]:
        """Filter stars specifically for constellation recognition."""
        logger.info("   Filtering stars for constellation recognition...")
        
        if len(stars) <= self.max_stars_for_matching:
            logger.info(f"   Using all {len(stars)} stars (under limit)")
            return stars
        
        # Sort by brightness and area (brightest, largest first)
        sorted_stars = sorted(stars, key=lambda s: (s["brightness"], s["area"]), reverse=True)
        
        # Take top stars for constellation matching
        filtered_stars = sorted_stars[:self.max_stars_for_matching]
        
        logger.info(f"   Filtered from {len(stars)} to {len(filtered_stars)} stars for constellation matching")
        
        return filtered_stars
    
    def _create_optimized_star_field(self, star_field: np.ndarray, stars: List[Dict]) -> np.ndarray:
        """Create optimized star field image."""
        # Create background that preserves some nebulosity
        result = star_field.copy()  # **PRESERVE** some background nebulosity
        
        # Draw detected stars with enhanced visualization
        for star in tqdm(stars, desc="   Drawing optimized stars"):
            x, y = star["x"], star["y"]
            brightness = int(star["brightness"])
            size = max(2, int(star["area"] / 10))
            
            cv2.circle(result, (x, y), size, brightness, -1)
        
        return result
    
    def estimate_optimized_field_of_view(self, stars: List[Dict], image_shape: Tuple[int, int]) -> Dict:
        """Estimate field of view with optimized analysis."""
        logger.info("🔍 Estimating optimized field of view...")
        
        if len(stars) < 5:
            logger.warning("   Warning: Too few stars for accurate FOV estimation")
            return {"fov_estimate": "unknown", "confidence": "low"}
        
        # Calculate star density
        image_area = image_shape[0] * image_shape[1]
        star_density = len(stars) / image_area
        
        # Analyze star brightness distribution
        brightnesses = [star["brightness"] for star in stars]
        avg_brightness = np.mean(brightnesses)
        
        # Analyze star spacing
        spacings = self._calculate_optimized_star_spacings(stars)
        avg_spacing = np.mean(spacings) if spacings else 0
        
        # Optimized FOV estimation
        fov_estimate = self._optimized_fov_estimation(star_density, avg_brightness, avg_spacing, image_shape)
        
        logger.info(f"   Star density: {star_density:.6f} stars/pixel²")
        logger.info(f"   Average brightness: {avg_brightness:.1f}")
        logger.info(f"   Average spacing: {avg_spacing:.1f} pixels")
        logger.info(f"   Estimated FOV: {fov_estimate}")
        
        return {
            "fov_estimate": fov_estimate,
            "star_density": star_density,
            "avg_brightness": avg_brightness,
            "avg_spacing": avg_spacing,
            "num_stars": len(stars),
            "confidence": "high" if len(stars) > 50 else "medium"
        }
    
    def _calculate_optimized_star_spacings(self, stars: List[Dict]) -> List[float]:
        """Calculate star spacings with optimized algorithm."""
        spacings = []
        
        # Use fewer stars for faster calculation
        sample_stars = stars[:100] if len(stars) > 100 else stars
        
        for i, star1 in enumerate(tqdm(sample_stars, desc="   Calculating optimized spacings")):
            for j, star2 in enumerate(sample_stars[i+1:], i+1):
                dx = star1["x"] - star2["x"]
                dy = star1["y"] - star2["y"]
                distance = np.sqrt(dx*dx + dy*dy)
                
                if distance < 200:  # Reasonable range for constellation spacing
                    spacings.append(distance)
        
        return spacings
    
    def _optimized_fov_estimation(self, density: float, brightness: float, spacing: float, 
                                 image_shape: Tuple[int, int]) -> str:
        """Optimized FOV estimation with adjusted thresholds."""
        width, height = image_shape[1], image_shape[0]
        
        # Adjusted thresholds for optimized detection
        if density > 0.0001 and brightness > 200:
            return "wide_field (60-90°)"
        elif density > 0.00005 and brightness > 170:
            return "medium_field (20-60°)"
        elif density > 0.00001 and brightness > 140:
            return "narrow_field (5-20°)"
        elif density < 0.00001:
            return "very_narrow_field (1-5°)"
        else:
            return "medium_field (20-60°)"  # Default to medium field
    
    def match_constellations_optimized(self, stars: List[Dict]) -> List[Dict]:
        """Match constellations with optimized parameters."""
        logger.info("🎯 Optimized constellation matching...")
        
        if not stars:
            raise ValueError("No stars available for matching")
        
        # Create multiple WCS configurations with optimized scales
        wcs_configs = [
            # Southern sky configurations with optimized scales
            {"name": "Southern Crux Region", "ra": 190, "dec": -62, "scale": 0.05},
            {"name": "Southern Carina Region", "ra": 140, "dec": -55, "scale": 0.08},
            {"name": "Southern Centaurus Region", "ra": 210, "dec": -60, "scale": 0.06},
            
            # Northern sky configurations
            {"name": "Northern Ursa Major", "ra": 165, "dec": 55, "scale": 0.08},
            {"name": "Northern Orion", "ra": 85, "dec": 5, "scale": 0.06},
            {"name": "Northern Cassiopeia", "ra": 15, "dec": 60, "scale": 0.10},
        ]
        
        best_matches = []
        best_config = None
        best_score = 0
        
        for config in tqdm(wcs_configs, desc="   Testing optimized WCS configurations"):
            logger.info(f"   Testing {config['name']}...")
            
            # Create WCS for this configuration
            wcs = self._create_wcs_from_config(config)
            
            # Try to match constellations with this WCS
            matches = self._match_constellations_with_wcs(wcs, stars)
            
            # Calculate total score for this configuration
            total_score = sum(match["confidence"] for match in matches)
            
            logger.info(f"   {config['name']}: {len(matches)} matches, score {total_score:.2f}")
            
            if total_score > best_score:
                best_score = total_score
                best_matches = matches
                best_config = config
        
        logger.info(f"   Best configuration: {best_config['name'] if best_config else 'None'}")
        logger.info(f"   Best score: {best_score:.2f}")
        logger.info(f"   Total matches: {len(best_matches)}")
        
        return best_matches
    
    def _create_wcs_from_config(self, config: Dict) -> WCS:
        """Create WCS from configuration."""
        height, width = 3413, 5120  # Image dimensions
        
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [width//2, height//2]  # Reference pixel at center
        wcs.wcs.crval = [config["ra"], config["dec"]]  # Center coordinates
        wcs.wcs.cdelt = [config["scale"], config["scale"]]  # Scale
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        
        return wcs
    
    def _match_constellations_with_wcs(self, wcs: WCS, stars: List[Dict]) -> List[Dict]:
        """Match constellations using a specific WCS with optimized parameters."""
        matches = []
        constellations = self.constellation_database.get("constellations", {})
        
        for const_name, const_data in constellations.items():
            # Get constellation stars
            const_stars = const_data.get("stars", [])
            if len(const_stars) < 2:
                continue
            
            # Try to match this constellation
            match_result = self._match_single_constellation_optimized(
                const_name, const_stars, const_data, wcs, stars
            )
            
            if match_result:
                matches.append(match_result)
        
        return matches
    
    def _match_single_constellation_optimized(self, const_name: str, const_stars: List[Dict], 
                                             const_data: Dict, wcs: WCS, stars: List[Dict]) -> Optional[Dict]:
        """Match a single constellation using WCS with optimized parameters."""
        
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
        
        if len(const_pixels) < 2:
            return None
        
        # Find matching detected stars with **OPTIMIZED** tolerance
        matched_stars = []
        total_score = 0
        
        for const_star in const_pixels:
            best_match = None
            best_distance = float('inf')
            
            # Find closest detected star within **OPTIMIZED** tolerance
            for detected_star in stars:
                distance = np.sqrt(
                    (const_star["pixel_x"] - detected_star["x"])**2 + 
                    (const_star["pixel_y"] - detected_star["y"])**2
                )
                
                # **OPTIMIZED** tolerance - more reasonable for real constellation matching
                if distance < best_distance and distance < 150:  # 150 pixel tolerance (was 300)
                    best_distance = distance
                    best_match = detected_star
            
            if best_match:
                matched_stars.append({
                    "constellation_star": const_star,
                    "detected_star": best_match,
                    "distance": best_distance
                })
                # **OPTIMIZED** scoring
                brightness_factor = best_match.get("brightness", 0) / 255.0
                distance_factor = 1.0 / (1.0 + best_distance / 75.0)  # Tighter distance factor
                total_score += brightness_factor * distance_factor
        
        if len(matched_stars) >= 2:
            confidence = total_score / len(const_pixels)
            
            # **OPTIMIZED** threshold - more reasonable for real matching
            if confidence > 0.15:  # Higher threshold (was 0.05)
                return {
                    "constellation": const_name,
                    "confidence": confidence,
                    "matched_stars": matched_stars,
                    "constellation_stars": const_pixels,
                    "constellation_data": const_data
                }
        
        return None
    
    def create_optimized_annotation(self, stars: List[Dict], matches: List[Dict], 
                                   fov_info: Dict, output_path: str = "Output/optimized_constellation_solution.jpg"):
        """Create optimized annotated image with constellation overlays."""
        logger.info("🎨 Creating optimized constellation annotation...")
        
        # Load original image
        original_image = cv2.imread("Input/test-1.jpg")
        if original_image is None:
            logger.error("❌ Could not load original image")
            return
        
        # Create annotation overlay
        annotated_image = original_image.copy()
        
        # Draw matched constellations
        for match in matches:
            self._draw_optimized_constellation(annotated_image, match)
        
        # Add comprehensive information
        annotated_image = self._add_optimized_info(annotated_image, stars, matches, fov_info)
        
        # Save result
        cv2.imwrite(output_path, annotated_image)
        logger.info(f"   ✅ Saved optimized annotation: {output_path}")
    
    def _draw_optimized_constellation(self, image: np.ndarray, match: Dict):
        """Draw a constellation with optimized visualization."""
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
                cv2.line(image, pt1, pt2, (0, 255, 255), 6)  # Thick yellow line
        
        # Draw stars and labels
        for matched_star in matched_stars:
            detected_star = matched_star["detected_star"]
            const_star = matched_star["constellation_star"]
            
            # Draw star with size based on brightness
            center = (int(detected_star["x"]), int(detected_star["y"]))
            brightness = detected_star.get("brightness", 0)
            size = max(4, int(brightness / 30))  # Larger stars for visibility
            
            cv2.circle(image, center, size, (255, 255, 255), -1)  # White star
            
            # Draw label
            label = f"{const_star['name']}"
            cv2.putText(image, label, (center[0] + 20, center[1] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Draw constellation name
        if matched_stars:
            center_x = int(np.mean([ms["detected_star"]["x"] for ms in matched_stars]))
            center_y = int(np.mean([ms["detected_star"]["y"] for ms in matched_stars]))
            
            label = f"{const_name} ({confidence:.2f})"
            cv2.putText(image, label, (center_x - 60, center_y - 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    
    def _add_optimized_info(self, image: np.ndarray, stars: List[Dict], matches: List[Dict], 
                           fov_info: Dict) -> np.ndarray:
        """Add optimized comprehensive information to the image."""
        # Convert to PIL for better text rendering
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 40)
            small_font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Add title
        title = "Optimized Constellation Recognition"
        draw.text((20, 20), title, fill=(255, 255, 255), font=font)
        
        # Add processing information
        y_offset = 100
        info_lines = [
            f"Stars detected: {len(stars)} (optimized for constellations)",
            f"FOV estimate: {fov_info.get('fov_estimate', 'unknown')}",
            f"Star density: {fov_info.get('star_density', 0):.6f} stars/pixel²",
            f"Constellations found: {len(matches)}",
            f"Min magnitude: {self.min_magnitude}",
            f"Max stars for matching: {self.max_stars_for_matching}"
        ]
        
        for line in info_lines:
            draw.text((20, y_offset), line, fill=(255, 255, 255), font=small_font)
            y_offset += 35
        
        # Add constellation details
        y_offset += 20
        for match in matches:
            const_name = match["constellation"]
            confidence = match["confidence"]
            num_stars = len(match["matched_stars"])
            
            line = f"• {const_name}: {confidence:.2f} ({num_stars} stars)"
            draw.text((20, y_offset), line, fill=(0, 255, 255), font=small_font)
            y_offset += 30
        
        # Add optimization details
        y_offset += 20
        draw.text((20, y_offset), "Optimizations applied:", 
                 fill=(255, 165, 0), font=small_font)  # Orange
        y_offset += 25
        draw.text((20, y_offset), "• Reduced sensitivity (higher magnitude threshold)", 
                 fill=(255, 165, 0), font=small_font)  # Orange
        y_offset += 25
        draw.text((20, y_offset), "• Limited stars for matching (500 max)", 
                 fill=(255, 165, 0), font=small_font)  # Orange
        y_offset += 25
        draw.text((20, y_offset), "• Better nebulosity preservation", 
                 fill=(255, 165, 0), font=small_font)  # Orange
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def main():
    """Test optimized constellation solving."""
    print("🎯 Optimized Constellation Solver Test")
    print("=" * 60)
    print("Testing reduced sensitivity and better nebulosity handling")
    print("=" * 60)
    
    start_time = time.time()
    
    # Create solver
    solver = OptimizedConstellationSolver()
    
    try:
        # Extract optimized star field
        star_field, stars = solver.extract_optimized_star_field("Input/test-1.jpg")
        
        # Estimate FOV
        fov_info = solver.estimate_optimized_field_of_view(stars, star_field.shape)
        
        # Save optimized star field
        cv2.imwrite("Processing/optimized_star_field.jpg", star_field)
        
        # Save optimized star data
        stars_serializable = []
        for star in stars:
            star_copy = star.copy()
            star_copy["brightness"] = float(star_copy["brightness"])
            star_copy["area"] = float(star_copy["area"])
            stars_serializable.append(star_copy)
        
        with open("Processing/optimized_detected_stars.json", 'w') as f:
            json.dump(stars_serializable, f, indent=2)
        
        # Save optimized FOV info
        with open("Processing/optimized_fov_estimation.json", 'w') as f:
            json.dump(fov_info, f, indent=2)
        
        # Match constellations
        matches = solver.match_constellations_optimized(stars)
        
        # Create optimized annotation
        solver.create_optimized_annotation(stars, matches, fov_info)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ Optimized constellation solving complete!")
        print(f"   Detected {len(stars)} stars (optimized for constellations)")
        print(f"   Found {len(matches)} constellation matches")
        print(f"   FOV estimate: {fov_info['fov_estimate']}")
        print(f"   Star density: {fov_info['star_density']:.6f} stars/pixel²")
        
        if matches:
            print("   Matched constellations:")
            for match in matches:
                print(f"     • {match['constellation']}: {match['confidence']:.2f} confidence")
        else:
            print("   No constellations matched - optimizations applied:")
            print("     • Higher magnitude threshold (150)")
            print("     • Limited stars for matching (500 max)")
            print("     • Better nebulosity preservation")
            print("     • More reasonable matching tolerance")
        
        print(f"   Total processing time: {elapsed_time:.1f} seconds")
        print(f"   Check Output/optimized_constellation_solution.jpg for results")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        raise

if __name__ == "__main__":
    main() 