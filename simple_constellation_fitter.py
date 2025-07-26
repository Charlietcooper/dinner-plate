#!/usr/bin/env python3
"""
Simple Constellation Fitter - Demonstrates constellation shape database concept
"""

import numpy as np
import cv2
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from typing import List, Dict, Tuple, Optional
import json
import os
import math

class SimpleStarDetector:
    """Simple star detection for demonstration."""
    
    def __init__(self):
        self.min_brightness = 30
        
    def detect_stars(self, image: np.ndarray) -> List[Dict]:
        """Detect bright stars in the image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Simple thresholding
        _, thresh = cv2.threshold(gray, self.min_brightness, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        stars = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 5 <= area <= 100:  # Filter by size
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    brightness = int(gray[cy, cx])
                    
                    stars.append({
                        "x": cx, "y": cy, "brightness": brightness, "area": area
                    })
        
        # Sort by brightness
        stars.sort(key=lambda x: x['brightness'], reverse=True)
        return stars[:50]  # Limit to top 50 brightest stars

class ConstellationShapeDatabase:
    """Database of constellation shapes with real astronomical coordinates."""
    
    def __init__(self):
        self.constellations = self._create_constellation_data()
    
    def _create_constellation_data(self) -> Dict:
        """Create constellation data with real coordinates."""
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

class SimpleConstellationFitter:
    """Simple constellation fitting demonstration."""
    
    def __init__(self, wcs: WCS):
        self.wcs = wcs
        self.star_detector = SimpleStarDetector()
        self.shape_db = ConstellationShapeDatabase()
    
    def analyze_image(self, image: np.ndarray) -> Dict:
        """Analyze image and show constellation fitting concept."""
        print("🔍 Analyzing image for constellation fitting...")
        
        # Detect stars
        detected_stars = self.star_detector.detect_stars(image)
        print(f"⭐ Detected {len(detected_stars)} bright stars")
        
        # Show constellation database concept
        constellations = self.shape_db.get_all_constellations()
        print(f"📊 Database contains {len(constellations)} constellation patterns")
        
        # Demonstrate coordinate conversion
        print("\n🌍 Demonstrating coordinate conversion:")
        for const_name, pattern in constellations.items():
            print(f"\n🔍 {const_name}:")
            visible_stars = 0
            
            for star in pattern['stars']:
                pixel = self._sky_to_pixel(star['ra'], star['dec'])
                if pixel:
                    visible_stars += 1
                    print(f"   {star['name']}: RA={star['ra']:.1f}°, Dec={star['dec']:.1f}° → Pixel({pixel[0]:.0f}, {pixel[1]:.0f})")
                else:
                    print(f"   {star['name']}: RA={star['ra']:.1f}°, Dec={star['dec']:.1f}° → Outside image")
            
            print(f"   📊 {visible_stars}/{len(pattern['stars'])} stars visible in image")
        
        # Show pattern matching concept
        print(f"\n🎯 Pattern Matching Concept:")
        print(f"   With real plate solving, the system would:")
        print(f"   1. Use Astrometry.net to get accurate WCS")
        print(f"   2. Convert constellation star positions to pixels")
        print(f"   3. Match detected stars to constellation patterns")
        print(f"   4. Handle scale variations (different focal lengths)")
        print(f"   5. Handle rotation (constellations rotate with time)")
        print(f"   6. Draw accurate constellation lines and labels")
        
        return {
            "detected_stars": len(detected_stars),
            "constellations": len(constellations),
            "wcs_accuracy": "Demo WCS (needs real plate solving)"
        }
    
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

def create_demonstration_image(image: np.ndarray, fitter: SimpleConstellationFitter) -> np.ndarray:
    """Create a demonstration image showing the concept with actual annotations."""
    # Create a copy for demonstration
    demo_image = image.copy()
    
    # Add explanatory text
    cv2.putText(demo_image, "Constellation Shape Database Demo", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    # Get constellation data and draw actual annotations
    constellations = fitter.shape_db.get_all_constellations()
    
    # Draw Ursa Major (Big Dipper) - the one we know is visible
    ursa_major = constellations["Ursa Major"]
    ursa_stars = {}
    
    # Convert stars to pixel coordinates
    for star in ursa_major["stars"]:
        pixel = fitter._sky_to_pixel(star["ra"], star["dec"])
        if pixel:
            ursa_stars[star["name"]] = pixel
    
    # Draw constellation lines
    if len(ursa_stars) >= 2:
        print(f"🎨 Drawing Ursa Major constellation lines...")
        
        for star1_name, star2_name in ursa_major["lines"]:
            if star1_name in ursa_stars and star2_name in ursa_stars:
                x1, y1 = int(ursa_stars[star1_name][0]), int(ursa_stars[star1_name][1])
                x2, y2 = int(ursa_stars[star2_name][0]), int(ursa_stars[star2_name][1])
                
                # Draw line
                cv2.line(demo_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Draw star markers
        for star_name, pixel in ursa_stars.items():
            x, y = int(pixel[0]), int(pixel[1])
            cv2.circle(demo_image, (x, y), 6, (255, 255, 0), -1)
            cv2.circle(demo_image, (x, y), 8, (255, 255, 0), 2)
        
        # Add constellation label
        center_x = int(sum(p[0] for p in ursa_stars.values()) / len(ursa_stars))
        center_y = int(sum(p[1] for p in ursa_stars.values()) / len(ursa_stars))
        
        cv2.putText(demo_image, "Ursa Major", (center_x + 20, center_y - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        print(f"✅ Drew Ursa Major with {len(ursa_stars)} stars")
    
    # Add some detected stars from the image
    detected_stars = fitter.star_detector.detect_stars(image)
    print(f"⭐ Drawing {min(20, len(detected_stars))} detected stars...")
    
    for i, star in enumerate(detected_stars[:20]):  # Show first 20 detected stars
        x, y = star["x"], star["y"]
        brightness = star["brightness"]
        
        # Draw small white circles for detected stars
        cv2.circle(demo_image, (x, y), 2, (255, 255, 255), -1)
    
    # Add legend
    cv2.putText(demo_image, "Legend:", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.putText(demo_image, "Green lines = Constellation connections", (50, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.putText(demo_image, "Yellow circles = Bright constellation stars", (50, 155), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    cv2.putText(demo_image, "White dots = Detected stars in image", (50, 180), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.putText(demo_image, "This shows how real constellation patterns would be fitted", (50, 210), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.putText(demo_image, "to your actual astronomical images!", (50, 235), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return demo_image

def main():
    """Demonstrate constellation shape fitting concept."""
    print("🌟 Constellation Shape Database Concept")
    print("=" * 50)
    
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
    fitter = SimpleConstellationFitter(wcs)
    
    # Analyze image
    analysis = fitter.analyze_image(image)
    
    # Create demonstration image
    demo_image = create_demonstration_image(image, fitter)
    
    # Save result
    output_path = "Output/constellation_concept_demo.jpg"
    success = cv2.imwrite(output_path, demo_image)
    
    if success:
        print(f"\n✅ Concept demonstration completed!")
        print(f"📸 Demo image saved as: {output_path}")
        print(f"📊 Analysis results:")
        print(f"   - Detected stars: {analysis['detected_stars']}")
        print(f"   - Constellation patterns: {analysis['constellations']}")
        print(f"   - WCS status: {analysis['wcs_accuracy']}")
        
        print(f"\n🎯 Next Steps for Real Implementation:")
        print(f"   1. Integrate with Astrometry.net for real plate solving")
        print(f"   2. Expand constellation database with more patterns")
        print(f"   3. Implement robust pattern matching algorithm")
        print(f"   4. Add scale and rotation handling")
        print(f"   5. Create accurate constellation line drawing")
        
        return True
    else:
        print("❌ Failed to save demo image")
        return False

if __name__ == "__main__":
    main() 