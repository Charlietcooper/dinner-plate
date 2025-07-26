#!/usr/bin/env python3
"""
Canonical Shape Demo - Shows how constellation shapes are preserved
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

class CanonicalShapeDemo:
    """Demonstrates canonical constellation shapes."""
    
    def __init__(self):
        self.canonical_shapes = self._create_canonical_shapes()
    
    def _create_canonical_shapes(self) -> Dict:
        """Create canonical constellation shapes with real coordinates."""
        return {
            "Crux": {
                "description": "Southern Cross - Canonical Shape",
                "stars": [
                    {"name": "Acrux", "ra": 186.6495, "dec": -63.0991, "mag": 0.77},
                    {"name": "Mimosa", "ra": 191.9303, "dec": -59.6888, "mag": 1.25},
                    {"name": "Gacrux", "ra": 187.7915, "dec": -57.1138, "mag": 1.59},
                    {"name": "Delta Crucis", "ra": 183.7863, "dec": -58.7489, "mag": 2.79}
                ],
                "lines": [
                    ("Acrux", "Mimosa"),
                    ("Mimosa", "Gacrux"),
                    ("Gacrux", "Delta Crucis"),
                    ("Delta Crucis", "Acrux")
                ],
                "canonical_shape": "Cross pattern with specific proportions"
            },
            "Carina": {
                "description": "The Keel - Canonical Shape",
                "stars": [
                    {"name": "Canopus", "ra": 95.9879, "dec": -52.6957, "mag": -0.74},
                    {"name": "Miaplacidus", "ra": 138.2999, "dec": -69.7172, "mag": 1.67},
                    {"name": "Avior", "ra": 139.2725, "dec": -59.5092, "mag": 1.86},
                    {"name": "Aspidiske", "ra": 139.2725, "dec": -59.5092, "mag": 2.21}
                ],
                "lines": [
                    ("Canopus", "Miaplacidus"),
                    ("Miaplacidus", "Avior"),
                    ("Avior", "Aspidiske")
                ],
                "canonical_shape": "Elongated keel shape with specific angles"
            }
        }
    
    def analyze_canonical_shapes(self) -> Dict:
        """Analyze the canonical shapes and their properties."""
        print("🔍 Analyzing canonical constellation shapes...")
        
        analysis = {}
        
        for const_name, shape in self.canonical_shapes.items():
            print(f"\n📐 {const_name}:")
            print(f"   Description: {shape['description']}")
            print(f"   Canonical Shape: {shape['canonical_shape']}")
            
            # Calculate shape properties
            coords = np.array([[s["ra"], s["dec"]] for s in shape["stars"]])
            
            # Calculate distances between stars
            distances = []
            for i in range(len(coords)):
                for j in range(i+1, len(coords)):
                    dist = np.sqrt(np.sum((coords[i] - coords[j])**2))
                    distances.append(dist)
                    print(f"   Distance {shape['stars'][i]['name']}-{shape['stars'][j]['name']}: {dist:.2f}°")
            
            # Calculate angles
            angles = []
            for i in range(len(coords)):
                for j in range(len(coords)):
                    if i == j:
                        continue
                    for k in range(len(coords)):
                        if k == i or k == j:
                            continue
                        
                        # Calculate angle at star j
                        v1 = coords[i] - coords[j]
                        v2 = coords[k] - coords[j]
                        
                        # Normalize vectors
                        v1_norm = v1 / np.linalg.norm(v1)
                        v2_norm = v2 / np.linalg.norm(v2)
                        
                        # Calculate angle
                        cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
                        angle = np.arccos(cos_angle) * 180 / np.pi
                        angles.append(angle)
            
            print(f"   Average distance: {np.mean(distances):.2f}°")
            print(f"   Distance range: {np.min(distances):.2f}° - {np.max(distances):.2f}°")
            print(f"   Average angle: {np.mean(angles):.1f}°")
            
            analysis[const_name] = {
                "distances": distances,
                "angles": angles,
                "coords": coords,
                "shape_properties": {
                    "avg_distance": np.mean(distances),
                    "distance_range": (np.min(distances), np.max(distances)),
                    "avg_angle": np.mean(angles)
                }
            }
        
        return analysis
    
    def demonstrate_shape_preservation(self, image: np.ndarray) -> np.ndarray:
        """Demonstrate how canonical shapes would be preserved."""
        demo_image = image.copy()
        
        # Add title
        cv2.putText(demo_image, "Canonical Constellation Shape Preservation", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Create demo WCS for southern sky
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [image.shape[1]//2, image.shape[0]//2]
        wcs.wcs.crval = [180, -60]  # Southern sky
        wcs.wcs.cdelt = [0.1, 0.1]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        
        # Draw canonical shapes
        for const_name, shape in self.canonical_shapes.items():
            print(f"\n🎨 Drawing canonical shape for {const_name}...")
            
            # Convert stars to pixel coordinates
            const_stars = {}
            for star in shape['stars']:
                pixel = self._sky_to_pixel(wcs, star['ra'], star['dec'])
                if pixel:
                    const_stars[star['name']] = pixel
            
            if len(const_stars) >= 2:
                # Draw constellation lines (canonical shape)
                for star1_name, star2_name in shape['lines']:
                    if star1_name in const_stars and star2_name in const_stars:
                        x1, y1 = int(const_stars[star1_name][0]), int(const_stars[star1_name][1])
                        x2, y2 = int(const_stars[star2_name][0]), int(const_stars[star2_name][1])
                        
                        # Draw canonical line
                        cv2.line(demo_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Draw star markers
                for star_name, pixel in const_stars.items():
                    x, y = int(pixel[0]), int(pixel[1])
                    cv2.circle(demo_image, (x, y), 6, (255, 255, 0), -1)
                    cv2.circle(demo_image, (x, y), 8, (255, 255, 0), 2)
                
                # Add constellation label
                center_x = int(sum(p[0] for p in const_stars.values()) / len(const_stars))
                center_y = int(sum(p[1] for p in const_stars.values()) / len(const_stars))
                
                cv2.putText(demo_image, f"{const_name} (Canonical)", (center_x + 20, center_y - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Add explanation
        cv2.putText(demo_image, "Canonical Shape Properties:", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(demo_image, "1. Real astronomical coordinates (RA/Dec)", (50, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(demo_image, "2. Preserved star distances and angles", (50, 155), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(demo_image, "3. Distortion limits (30% distance, 15° angle)", (50, 180), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(demo_image, "4. Shape preservation during fitting", (50, 205), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(demo_image, "5. Real plate solving provides accurate WCS", (50, 230), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return demo_image
    
    def _sky_to_pixel(self, wcs: WCS, ra: float, dec: float) -> Optional[Tuple[float, float]]:
        """Convert sky coordinates to pixel coordinates."""
        try:
            coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
            x, y = wcs.world_to_pixel(coord)
            if not (np.isnan(x) or np.isnan(y)):
                return float(x), float(y)
        except:
            pass
        return None

def main():
    """Demonstrate canonical constellation shapes."""
    print("🌟 Canonical Constellation Shape Demo")
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
    
    # Create demo
    demo = CanonicalShapeDemo()
    
    # Analyze canonical shapes
    analysis = demo.analyze_canonical_shapes()
    
    # Create demonstration image
    demo_image = demo.demonstrate_shape_preservation(image)
    
    # Save result
    output_path = "Output/canonical_shape_demo.jpg"
    success = cv2.imwrite(output_path, demo_image)
    
    if success:
        print(f"\n✅ Canonical shape demo completed!")
        print(f"📸 Demo image saved as: {output_path}")
        
        print(f"\n📊 Canonical Shape Analysis:")
        for const_name, data in analysis.items():
            print(f"   {const_name}:")
            print(f"     - Average distance: {data['shape_properties']['avg_distance']:.2f}°")
            print(f"     - Distance range: {data['shape_properties']['distance_range'][0]:.2f}° - {data['shape_properties']['distance_range'][1]:.2f}°")
            print(f"     - Average angle: {data['shape_properties']['avg_angle']:.1f}°")
        
        print(f"\n🎯 Key Points:")
        print(f"   1. Canonical shapes have specific distance and angle relationships")
        print(f"   2. Shape preservation ensures constellations look recognizable")
        print(f"   3. Distortion limits prevent unrealistic deformations")
        print(f"   4. Real plate solving would provide accurate positioning")
        print(f"   5. Pattern matching finds the best fit within shape constraints")
        
        return True
    else:
        print("❌ Failed to save demo image")
        return False

if __name__ == "__main__":
    main() 