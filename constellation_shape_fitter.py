#!/usr/bin/env python3
"""
Constellation Shape Fitter - Advanced constellation detection using pattern matching
"""

import numpy as np
import cv2
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from typing import List, Dict, Tuple, Optional
import json
import os

class StarDetector:
    """Detect stars in astronomical images."""
    
    def __init__(self):
        self.min_star_brightness = 30
        self.max_star_brightness = 255
        
    def detect_stars(self, image: np.ndarray) -> List[Dict]:
        """Detect stars in the image using blob detection."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Find bright spots (stars)
        stars = []
        height, width = gray.shape
        
        # Use thresholding to find bright objects
        _, thresh = cv2.threshold(blurred, self.min_star_brightness, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Calculate contour properties
            area = cv2.contourArea(contour)
            if area < 5 or area > 100:  # Filter by size
                continue
                
            # Get centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Get brightness at this point
                brightness = int(gray[cy, cx])
                
                if self.min_star_brightness <= brightness <= self.max_star_brightness:
                    stars.append({
                        "x": cx,
                        "y": cy,
                        "brightness": brightness,
                        "area": area
                    })
        
        return stars

class ConstellationShapeDatabase:
    """Database of constellation shapes and patterns."""
    
    def __init__(self):
        self.constellations = self._load_constellation_data()
    
    def _load_constellation_data(self) -> Dict:
        """Load constellation shape data from JSON file or create default."""
        data_file = "constellation_shapes.json"
        
        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                return json.load(f)
        else:
            # Create default constellation data
            return self._create_default_constellations()
    
    def _create_default_constellations(self) -> Dict:
        """Create default constellation data with real astronomical coordinates."""
        return {
            "Ursa Major": {
                "description": "The Great Bear - Big Dipper",
                "stars": [
                    {"name": "Dubhe", "ra": 165.9319, "dec": 61.7511, "mag": 1.79},
                    {"name": "Merak", "ra": 165.4603, "dec": 56.3824, "mag": 2.37},
                    {"name": "Phecda", "ra": 178.4577, "dec": 53.6948, "mag": 2.44},
                    {"name": "Megrez", "ra": 183.8565, "dec": 57.0326, "mag": 3.32},
                    {"name": "Alioth", "ra": 193.5073, "dec": 55.9598, "mag": 1.76},
                    {"name": "Mizar", "ra": 200.9814, "dec": 54.9254, "mag": 2.23},
                    {"name": "Alkaid", "ra": 206.8852, "dec": 49.3133, "mag": 1.85}
                ],
                "lines": [
                    ("Dubhe", "Merak"),
                    ("Merak", "Phecda"),
                    ("Phecda", "Megrez"),
                    ("Megrez", "Alioth"),
                    ("Alioth", "Mizar"),
                    ("Mizar", "Alkaid")
                ],
                "center_ra": 183.0,
                "center_dec": 55.0,
                "size_deg": 25.0
            },
            "Orion": {
                "description": "The Hunter",
                "stars": [
                    {"name": "Betelgeuse", "ra": 88.7929, "dec": 7.4071, "mag": 0.42},
                    {"name": "Bellatrix", "ra": 81.2828, "dec": 6.3497, "mag": 1.64},
                    {"name": "Mintaka", "ra": 83.0016, "dec": -0.2991, "mag": 2.25},
                    {"name": "Alnilam", "ra": 84.0534, "dec": -1.2019, "mag": 1.69},
                    {"name": "Alnitak", "ra": 85.1897, "dec": -1.9426, "mag": 1.88},
                    {"name": "Saiph", "ra": 86.9391, "dec": -9.6696, "mag": 2.07},
                    {"name": "Rigel", "ra": 78.6345, "dec": -8.2016, "mag": 0.18}
                ],
                "lines": [
                    ("Betelgeuse", "Bellatrix"),
                    ("Bellatrix", "Mintaka"),
                    ("Mintaka", "Alnilam"),
                    ("Alnilam", "Alnitak"),
                    ("Alnitak", "Saiph"),
                    ("Saiph", "Rigel"),
                    ("Rigel", "Betelgeuse")
                ],
                "center_ra": 83.0,
                "center_dec": -1.0,
                "size_deg": 20.0
            },
            "Cassiopeia": {
                "description": "The Queen - W-shaped constellation",
                "stars": [
                    {"name": "Schedar", "ra": 10.1268, "dec": 56.5373, "mag": 2.24},
                    {"name": "Caph", "ra": 2.2945, "dec": 59.1498, "mag": 2.28},
                    {"name": "Gamma Cas", "ra": 14.1772, "dec": 60.7167, "mag": 2.15},
                    {"name": "Ruchbah", "ra": 21.4538, "dec": 60.2353, "mag": 2.68},
                    {"name": "Segin", "ra": 28.5988, "dec": 63.6701, "mag": 3.35}
                ],
                "lines": [
                    ("Schedar", "Caph"),
                    ("Caph", "Gamma Cas"),
                    ("Gamma Cas", "Ruchbah"),
                    ("Ruchbah", "Segin")
                ],
                "center_ra": 15.0,
                "center_dec": 60.0,
                "size_deg": 15.0
            }
        }
    
    def get_constellation_pattern(self, name: str) -> Optional[Dict]:
        """Get constellation pattern by name."""
        return self.constellations.get(name)
    
    def get_all_constellations(self) -> Dict:
        """Get all constellation patterns."""
        return self.constellations

class ConstellationFitter:
    """Fit constellation patterns to detected stars."""
    
    def __init__(self, wcs: WCS):
        self.wcs = wcs
        self.star_detector = StarDetector()
        self.shape_db = ConstellationShapeDatabase()
    
    def fit_constellation(self, image: np.ndarray, constellation_name: str) -> Optional[Dict]:
        """Fit a specific constellation to the image."""
        # Get constellation pattern
        pattern = self.shape_db.get_constellation_pattern(constellation_name)
        if not pattern:
            return None
        
        # Detect stars in image
        detected_stars = self.star_detector.detect_stars(image)
        if len(detected_stars) < 3:
            return None
        
        # Convert constellation stars to pixel coordinates
        pattern_stars = []
        for star in pattern["stars"]:
            pixel = self._sky_to_pixel(star["ra"], star["dec"])
            if pixel:
                pattern_stars.append({
                    "name": star["name"],
                    "x": pixel[0],
                    "y": pixel[1],
                    "brightness": star["mag"]
                })
        
        if len(pattern_stars) < 3:
            return None
        
        # Try to match pattern to detected stars
        matches = self._match_pattern(pattern_stars, detected_stars)
        
        if matches:
            return {
                "constellation": constellation_name,
                "pattern_stars": pattern_stars,
                "detected_matches": matches,
                "lines": pattern["lines"],
                "center": self._calculate_center(matches)
            }
        
        return None
    
    def _sky_to_pixel(self, ra: float, dec: float) -> Optional[Tuple[float, float]]:
        """Convert sky coordinates to pixel coordinates."""
        try:
            coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
            x, y = self.wcs.world_to_pixel(coord)
            if not (np.isnan(x) or np.isnan(y)):
                return float(x), float(y)
        except:
            pass
        return None
    
    def _match_pattern(self, pattern_stars: List[Dict], detected_stars: List[Dict]) -> Optional[List[Dict]]:
        """Match constellation pattern to detected stars."""
        # Simple matching based on proximity and brightness
        matches = []
        
        for pattern_star in pattern_stars:
            best_match = None
            best_distance = float('inf')
            
            for detected_star in detected_stars:
                # Calculate distance
                distance = np.sqrt((pattern_star["x"] - detected_star["x"])**2 + 
                                 (pattern_star["y"] - detected_star["y"])**2)
                
                # Check if within reasonable distance (50 pixels)
                if distance < 50 and distance < best_distance:
                    best_match = detected_star
                    best_distance = distance
            
            if best_match:
                matches.append({
                    "pattern_star": pattern_star,
                    "detected_star": best_match,
                    "distance": best_distance
                })
        
        # Require at least 3 matches for a valid fit
        if len(matches) >= 3:
            return matches
        
        return None
    
    def _calculate_center(self, matches: List[Dict]) -> Tuple[float, float]:
        """Calculate center of matched constellation."""
        x_sum = sum(m["detected_star"]["x"] for m in matches)
        y_sum = sum(m["detected_star"]["y"] for m in matches)
        count = len(matches)
        
        return x_sum / count, y_sum / count

def main():
    """Test the constellation shape fitting."""
    print("🌟 Constellation Shape Fitter Test")
    print("=" * 40)
    
    # Load test image
    image_path = "Input/test-1.jpg"
    if not os.path.exists(image_path):
        print(f"❌ Test image not found: {image_path}")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return
    
    print(f"📸 Loaded image: {image.shape[1]} x {image.shape[0]} pixels")
    
    # Create demo WCS (in real usage, this would come from plate solving)
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [image.shape[1]//2, image.shape[0]//2]
    wcs.wcs.crval = [180, 0]
    wcs.wcs.cdelt = [0.1, 0.1]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    
    # Create fitter
    fitter = ConstellationFitter(wcs)
    
    # Test fitting constellations
    test_constellations = ["Ursa Major", "Orion", "Cassiopeia"]
    
    for const_name in test_constellations:
        print(f"\n🔍 Testing {const_name}...")
        result = fitter.fit_constellation(image, const_name)
        
        if result:
            print(f"✅ Found {const_name} with {len(result['detected_matches'])} matches")
            print(f"   Center: {result['center']}")
        else:
            print(f"❌ {const_name} not found in image")

if __name__ == "__main__":
    main() 