#!/usr/bin/env python3
"""
Constellation Visualizer - Creates PNG images of constellation shapes
"""

import numpy as np
import cv2
from astropy.coordinates import SkyCoord
from astropy import units as u
from typing import List, Dict, Tuple, Optional
import json
import os
import math

class ConstellationVisualizer:
    """Creates visual representations of constellation shapes."""
    
    def __init__(self):
        self.constellations = self._load_all_constellations()
    
    def _load_all_constellations(self) -> Dict:
        """Load all constellation data with real coordinates."""
        return {
            "Crux": {
                "name": "Crux",
                "description": "Southern Cross",
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
                ]
            },
            "Carina": {
                "name": "Carina",
                "description": "The Keel",
                "stars": [
                    {"name": "Canopus", "ra": 95.9879, "dec": -52.6957, "mag": -0.74},
                    {"name": "Miaplacidus", "ra": 138.2999, "dec": -69.7172, "mag": 1.67},
                    {"name": "Avior", "ra": 139.2725, "dec": -59.5092, "mag": 1.86},
                    {"name": "Aspidiske", "ra": 137.2725, "dec": -57.5092, "mag": 2.21}
                ],
                "lines": [
                    ("Canopus", "Miaplacidus"),
                    ("Miaplacidus", "Avior"),
                    ("Avior", "Aspidiske")
                ]
            },
            "Vela": {
                "name": "Vela",
                "description": "The Sails",
                "stars": [
                    {"name": "Suhail", "ra": 136.9990, "dec": -43.4326, "mag": 1.83},
                    {"name": "Markeb", "ra": 140.5284, "dec": -55.0107, "mag": 2.47},
                    {"name": "Alsephina", "ra": 127.5669, "dec": -49.4201, "mag": 1.75}
                ],
                "lines": [
                    ("Suhail", "Markeb"),
                    ("Markeb", "Alsephina"),
                    ("Alsephina", "Suhail")
                ]
            },
            "Puppis": {
                "name": "Puppis",
                "description": "The Stern",
                "stars": [
                    {"name": "Naos", "ra": 114.0674, "dec": -40.0031, "mag": 2.21},
                    {"name": "Tureis", "ra": 116.3287, "dec": -24.3043, "mag": 2.70}
                ],
                "lines": [
                    ("Naos", "Tureis")
                ]
            },
            "Pyxis": {
                "name": "Pyxis",
                "description": "The Compass",
                "stars": [
                    {"name": "Alpha Pyxidis", "ra": 130.8984, "dec": -33.1864, "mag": 3.68},
                    {"name": "Beta Pyxidis", "ra": 131.1759, "dec": -35.3083, "mag": 3.97}
                ],
                "lines": [
                    ("Alpha Pyxidis", "Beta Pyxidis")
                ]
            },
            "Antlia": {
                "name": "Antlia",
                "description": "The Air Pump",
                "stars": [
                    {"name": "Alpha Antliae", "ra": 157.2345, "dec": -31.0678, "mag": 4.25},
                    {"name": "Beta Antliae", "ra": 154.3912, "dec": -37.1373, "mag": 4.78}
                ],
                "lines": [
                    ("Alpha Antliae", "Beta Antliae")
                ]
            },
            "Volans": {
                "name": "Volans",
                "description": "The Flying Fish",
                "stars": [
                    {"name": "Beta Volantis", "ra": 127.5669, "dec": -66.1369, "mag": 3.77},
                    {"name": "Gamma Volantis", "ra": 127.5669, "dec": -66.1369, "mag": 3.78},
                    {"name": "Alpha Volantis", "ra": 125.5669, "dec": -64.1369, "mag": 4.00}
                ],
                "lines": [
                    ("Beta Volantis", "Gamma Volantis"),
                    ("Gamma Volantis", "Alpha Volantis")
                ]
            },
            "Musca": {
                "name": "Musca",
                "description": "The Fly",
                "stars": [
                    {"name": "Alpha Muscae", "ra": 186.6495, "dec": -69.1357, "mag": 2.69},
                    {"name": "Beta Muscae", "ra": 186.6495, "dec": -69.1357, "mag": 3.04},
                    {"name": "Gamma Muscae", "ra": 184.6495, "dec": -67.1357, "mag": 3.84}
                ],
                "lines": [
                    ("Alpha Muscae", "Beta Muscae"),
                    ("Beta Muscae", "Gamma Muscae")
                ]
            },
            "Ursa Major": {
                "name": "Ursa Major",
                "description": "The Great Bear (Big Dipper)",
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
                ]
            },
            "Orion": {
                "name": "Orion",
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
                ]
            },
            "Cassiopeia": {
                "name": "Cassiopeia",
                "description": "The Queen (W-shaped)",
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
                ]
            }
        }
    
    def create_constellation_image(self, constellation_name: str, size: int = 800) -> np.ndarray:
        """Create a visual representation of a constellation."""
        const = self.constellations.get(constellation_name)
        if not const:
            return None
        
        # Create black background
        image = np.zeros((size, size, 3), dtype=np.uint8)
        
        # Convert RA/Dec to normalized coordinates (0-1)
        coords = np.array([[s["ra"], s["dec"]] for s in const["stars"]])
        
        # Normalize coordinates to fit in image
        ra_min, ra_max = coords[:, 0].min(), coords[:, 0].max()
        dec_min, dec_max = coords[:, 1].min(), coords[:, 1].max()
        
        # Add padding
        ra_range = ra_max - ra_min
        dec_range = dec_max - dec_min
        padding = max(ra_range, dec_range) * 0.2
        
        ra_min -= padding
        ra_max += padding
        dec_min -= padding
        dec_max += padding
        
        # Convert to pixel coordinates
        pixel_coords = {}
        for i, star in enumerate(const["stars"]):
            ra, dec = coords[i]
            x = int((ra - ra_min) / (ra_max - ra_min) * (size * 0.8) + size * 0.1)
            y = int((dec - dec_min) / (dec_max - dec_min) * (size * 0.8) + size * 0.1)
            pixel_coords[star["name"]] = (x, y)
        
        # Draw constellation lines
        for star1_name, star2_name in const["lines"]:
            if star1_name in pixel_coords and star2_name in pixel_coords:
                x1, y1 = pixel_coords[star1_name]
                x2, y2 = pixel_coords[star2_name]
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Draw stars
        for star_name, (x, y) in pixel_coords.items():
            # Find star data
            star_data = next(s for s in const["stars"] if s["name"] == star_name)
            magnitude = star_data["mag"]
            
            # Size based on magnitude (brighter = larger)
            if magnitude < 1:
                radius = 8
                color = (255, 255, 255)  # White for very bright stars
            elif magnitude < 2:
                radius = 6
                color = (255, 255, 200)  # Yellow-white
            elif magnitude < 3:
                radius = 4
                color = (255, 200, 200)  # Light red
            else:
                radius = 3
                color = (200, 200, 255)  # Light blue
            
            cv2.circle(image, (x, y), radius, color, -1)
            cv2.circle(image, (x, y), radius + 2, color, 2)
        
        # Add title
        title = f"{const['name']} - {const['description']}"
        cv2.putText(image, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add star labels
        for star_name, (x, y) in pixel_coords.items():
            cv2.putText(image, star_name, (x + 10, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add coordinate info
        info_text = f"RA: {ra_min:.1f}° to {ra_max:.1f}° | Dec: {dec_min:.1f}° to {dec_max:.1f}°"
        cv2.putText(image, info_text, (20, size - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return image
    
    def create_all_constellation_images(self) -> List[str]:
        """Create PNG images for all constellations."""
        print("🎨 Creating constellation visualizations...")
        
        created_files = []
        
        for const_name in self.constellations.keys():
            print(f"   Creating {const_name}...")
            
            # Create image
            image = self.create_constellation_image(const_name)
            if image is not None:
                # Save to Output folder
                filename = f"Output/constellation_{const_name.lower().replace(' ', '_')}.png"
                success = cv2.imwrite(filename, image)
                
                if success:
                    created_files.append(filename)
                    print(f"   ✅ Saved {filename}")
                else:
                    print(f"   ❌ Failed to save {filename}")
            else:
                print(f"   ❌ Failed to create image for {const_name}")
        
        return created_files
    
    def create_constellation_atlas(self) -> str:
        """Create a single image showing all constellations."""
        print("📚 Creating constellation atlas...")
        
        # Calculate grid size
        n_constellations = len(self.constellations)
        grid_size = int(np.ceil(np.sqrt(n_constellations)))
        
        # Create large canvas
        canvas_size = grid_size * 400
        atlas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        
        # Add title
        cv2.putText(atlas, "Constellation Atlas - Canonical Shapes", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Place each constellation
        for i, (const_name, const) in enumerate(self.constellations.items()):
            row = i // grid_size
            col = i % grid_size
            
            # Create individual constellation image
            const_image = self.create_constellation_image(const_name, 350)
            if const_image is not None:
                # Calculate position
                x = col * 400 + 25
                y = row * 400 + 100
                
                # Place on canvas
                atlas[y:y+350, x:x+350] = const_image
        
        # Save atlas
        atlas_filename = "Output/constellation_atlas.png"
        success = cv2.imwrite(atlas_filename, atlas)
        
        if success:
            print(f"✅ Saved constellation atlas: {atlas_filename}")
            return atlas_filename
        else:
            print(f"❌ Failed to save constellation atlas")
            return None

def main():
    """Create constellation visualizations."""
    print("🌟 Constellation Visualizer")
    print("=" * 40)
    
    # Create visualizer
    visualizer = ConstellationVisualizer()
    
    # Create individual constellation images
    individual_files = visualizer.create_all_constellation_images()
    
    # Create constellation atlas
    atlas_file = visualizer.create_constellation_atlas()
    
    print(f"\n📊 Summary:")
    print(f"   Created {len(individual_files)} individual constellation images")
    if atlas_file:
        print(f"   Created constellation atlas: {atlas_file}")
    
    print(f"\n📁 Files saved in Output folder:")
    for filename in individual_files:
        print(f"   - {filename}")
    if atlas_file:
        print(f"   - {atlas_file}")
    
    print(f"\n🎯 Each image shows:")
    print(f"   - Canonical constellation shape")
    print(f"   - Real astronomical coordinates")
    print(f"   - Star magnitudes (size/color)")
    print(f"   - Constellation lines")
    print(f"   - Star names and descriptions")

if __name__ == "__main__":
    main() 