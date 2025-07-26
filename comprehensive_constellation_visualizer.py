#!/usr/bin/env python3
"""
Comprehensive Constellation Visualizer - Creates PNG images for all constellations
"""

import numpy as np
import cv2
import json
import os
from typing import Dict, List, Tuple, Optional

class ComprehensiveConstellationVisualizer:
    """Creates visual representations of all constellations from the comprehensive database."""
    
    def __init__(self, database_file: str = "Processing/comprehensive_constellation_database.json"):
        self.constellations = self._load_database(database_file)
    
    def _load_database(self, database_file: str) -> Dict:
        """Load constellation database from JSON file."""
        try:
            with open(database_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Failed to load database: {e}")
            return {}
    
    def create_constellation_image(self, constellation_name: str, size: int = 800) -> Optional[np.ndarray]:
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
            bayer = star_data.get("bayer", "")
            
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
        
        # Add hemisphere info
        hemisphere_text = f"Hemisphere: {const['hemisphere'].title()}"
        cv2.putText(image, hemisphere_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        # Add star labels
        for star_name, (x, y) in pixel_coords.items():
            star_data = next(s for s in const["stars"] if s["name"] == star_name)
            bayer = star_data.get("bayer", "")
            label = f"{star_name} ({bayer})" if bayer else star_name
            
            cv2.putText(image, label, (x + 10, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add coordinate info
        info_text = f"RA: {ra_min:.1f}° to {ra_max:.1f}° | Dec: {dec_min:.1f}° to {dec_max:.1f}°"
        cv2.putText(image, info_text, (20, size - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return image
    
    def create_all_constellation_images(self) -> List[str]:
        """Create PNG images for all constellations."""
        print("🎨 Creating comprehensive constellation visualizations...")
        
        created_files = []
        
        for const_name in self.constellations.keys():
            print(f"   Creating {const_name}...")
            
            # Create image
            image = self.create_constellation_image(const_name)
            if image is not None:
                # Save to Output folder
                filename = f"Output/comprehensive_{const_name.lower().replace(' ', '_')}.png"
                success = cv2.imwrite(filename, image)
                
                if success:
                    created_files.append(filename)
                    print(f"   ✅ Saved {filename}")
                else:
                    print(f"   ❌ Failed to save {filename}")
            else:
                print(f"   ❌ Failed to create image for {const_name}")
        
        return created_files
    
    def create_comprehensive_atlas(self) -> str:
        """Create a comprehensive atlas showing all constellations."""
        print("📚 Creating comprehensive constellation atlas...")
        
        # Calculate grid size
        n_constellations = len(self.constellations)
        grid_size = int(np.ceil(np.sqrt(n_constellations)))
        
        # Create large canvas
        canvas_size = grid_size * 400
        atlas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        
        # Add title
        cv2.putText(atlas, "Comprehensive Constellation Atlas - All 88 IAU Constellations", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
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
        atlas_filename = "Output/comprehensive_constellation_atlas.png"
        success = cv2.imwrite(atlas_filename, atlas)
        
        if success:
            print(f"✅ Saved comprehensive constellation atlas: {atlas_filename}")
            return atlas_filename
        else:
            print(f"❌ Failed to save comprehensive constellation atlas")
            return None
    
    def create_hemisphere_maps(self) -> Dict[str, str]:
        """Create separate maps for northern and southern hemispheres."""
        print("🌍 Creating hemisphere-specific constellation maps...")
        
        hemisphere_files = {}
        
        for hemisphere in ["northern", "southern"]:
            print(f"   Creating {hemisphere} hemisphere map...")
            
            # Get constellations for this hemisphere
            hemisphere_constellations = {name: const for name, const in self.constellations.items() 
                                       if const["hemisphere"] == hemisphere}
            
            if not hemisphere_constellations:
                continue
            
            # Calculate grid size
            n_constellations = len(hemisphere_constellations)
            grid_size = int(np.ceil(np.sqrt(n_constellations)))
            
            # Create canvas
            canvas_size = grid_size * 400
            hemisphere_map = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
            
            # Add title
            title = f"{hemisphere.title()} Hemisphere Constellations"
            cv2.putText(hemisphere_map, title, (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            
            # Place constellations
            for i, (const_name, const) in enumerate(hemisphere_constellations.items()):
                row = i // grid_size
                col = i % grid_size
                
                const_image = self.create_constellation_image(const_name, 350)
                if const_image is not None:
                    x = col * 400 + 25
                    y = row * 400 + 100
                    hemisphere_map[y:y+350, x:x+350] = const_image
            
            # Save hemisphere map
            filename = f"Output/{hemisphere}_hemisphere_constellations.png"
            success = cv2.imwrite(filename, hemisphere_map)
            
            if success:
                hemisphere_files[hemisphere] = filename
                print(f"   ✅ Saved {filename}")
            else:
                print(f"   ❌ Failed to save {filename}")
        
        return hemisphere_files

def main():
    """Create comprehensive constellation visualizations."""
    print("🌟 Comprehensive Constellation Visualizer")
    print("=" * 50)
    
    # Create visualizer
    visualizer = ComprehensiveConstellationVisualizer()
    
    if not visualizer.constellations:
        print("❌ No constellation data available")
        return
    
    # Create individual constellation images
    individual_files = visualizer.create_all_constellation_images()
    
    # Create comprehensive atlas
    atlas_file = visualizer.create_comprehensive_atlas()
    
    # Create hemisphere maps
    hemisphere_files = visualizer.create_hemisphere_maps()
    
    print(f"\n📊 Summary:")
    print(f"   Created {len(individual_files)} individual constellation images")
    if atlas_file:
        print(f"   Created comprehensive atlas: {atlas_file}")
    print(f"   Created {len(hemisphere_files)} hemisphere maps")
    
    print(f"\n📁 Files created in Output folder:")
    for filename in individual_files:
        print(f"   - {filename}")
    if atlas_file:
        print(f"   - {atlas_file}")
    for hemisphere, filename in hemisphere_files.items():
        print(f"   - {filename}")
    
    print(f"\n🎯 Key Features:")
    print(f"   - All constellations with real astronomical coordinates")
    print(f"   - Accurate star magnitudes and Bayer designations")
    print(f"   - Proper constellation line definitions")
    print(f"   - Hemisphere classification")
    print(f"   - Professional-grade visualizations")
    
    print(f"\n🌌 Based on Martin Krzywinski's constellation database structure")
    print(f"   from https://mk.bcgsc.ca/constellations/sky-constellations.mhtml")

if __name__ == "__main__":
    main() 