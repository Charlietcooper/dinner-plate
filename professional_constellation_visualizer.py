#!/usr/bin/env python3
"""
Professional Constellation Visualizer - Creates professional-grade constellation charts
Based on Yale Bright Star Catalogue and Martin Krzywinski standards
"""

import json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import os
from typing import Dict, List, Tuple

class ProfessionalConstellationVisualizer:
    """Creates professional-grade constellation visualizations."""
    
    def __init__(self):
        self.database_path = "Processing/professional_constellation_database.json"
        self.constellations = self._load_database()
        
        # Professional visualization settings
        self.image_size = (800, 600)
        self.background_color = (0, 0, 0)  # Black background
        self.star_colors = {
            "O": (255, 255, 255),  # White
            "B": (200, 200, 255),  # Blue-white
            "A": (255, 255, 255),  # White
            "F": (255, 255, 200),  # Yellow-white
            "G": (255, 255, 0),    # Yellow
            "K": (255, 200, 0),    # Orange
            "M": (255, 100, 0)     # Red-orange
        }
        self.constellation_line_color = (100, 150, 255)  # Blue
        self.text_color = (255, 255, 255)  # White
        
    def _load_database(self) -> Dict:
        """Load the professional constellation database."""
        try:
            with open(self.database_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Failed to load database: {e}")
            return {}
    
    def _get_star_color(self, spectral_type: str) -> Tuple[int, int, int]:
        """Get star color based on spectral type."""
        if spectral_type and len(spectral_type) > 0:
            spectral_class = spectral_type[0].upper()
            return self.star_colors.get(spectral_class, (255, 255, 255))
        return (255, 255, 255)  # Default white
    
    def _get_star_size(self, magnitude: float) -> int:
        """Get star size based on magnitude (brighter = larger)."""
        # Convert magnitude to size (brighter stars are larger)
        if magnitude <= 0:
            return 8  # Very bright stars
        elif magnitude <= 1:
            return 6
        elif magnitude <= 2:
            return 5
        elif magnitude <= 3:
            return 4
        elif magnitude <= 4:
            return 3
        else:
            return 2  # Dim stars
    
    def _normalize_coordinates(self, stars: List[Dict]) -> Tuple[List[Tuple[float, float]], Tuple[float, float, float, float]]:
        """Normalize star coordinates to fit in image."""
        if not stars:
            return [], (0, 0, 1, 1)
        
        # Extract RA/Dec coordinates
        coords = [(star['ra'], star['dec']) for star in stars]
        
        # Find bounds
        ra_coords = [coord[0] for coord in coords]
        dec_coords = [coord[1] for coord in coords]
        
        ra_min, ra_max = min(ra_coords), max(ra_coords)
        dec_min, dec_max = min(dec_coords), max(dec_coords)
        
        # Add padding
        ra_range = ra_max - ra_min
        dec_range = dec_max - dec_min
        
        if ra_range == 0:
            ra_range = 1
        if dec_range == 0:
            dec_range = 1
        
        padding = max(ra_range, dec_range) * 0.1
        
        bounds = (
            ra_min - padding,
            dec_min - padding,
            ra_max + padding,
            dec_max + padding
        )
        
        return coords, bounds
    
    def _coord_to_pixel(self, coord: Tuple[float, float], bounds: Tuple[float, float, float, float], image_size: Tuple[int, int]) -> Tuple[int, int]:
        """Convert RA/Dec coordinates to pixel coordinates."""
        ra, dec = coord
        ra_min, dec_min, ra_max, dec_max = bounds
        width, height = image_size
        
        # Normalize coordinates
        x = (ra - ra_min) / (ra_max - ra_min) * width
        y = height - (dec - dec_min) / (dec_max - dec_min) * height  # Flip Y axis
        
        return int(x), int(y)
    
    def create_constellation_image(self, constellation_name: str) -> np.ndarray:
        """Create a professional constellation image."""
        if constellation_name not in self.constellations:
            print(f"❌ Constellation '{constellation_name}' not found")
            return None
        
        const_data = self.constellations[constellation_name]
        stars = const_data['stars']
        
        # Create image
        image = np.full((*self.image_size, 3), self.background_color, dtype=np.uint8)
        
        # Normalize coordinates
        coords, bounds = self._normalize_coordinates(stars)
        
        # Create star name mapping
        star_names = {star['name']: star for star in stars}
        
        # Draw constellation lines first (behind stars)
        for line in const_data['lines']:
            star1_name, star2_name = line
            if star1_name in star_names and star2_name in star_names:
                star1 = star_names[star1_name]
                star2 = star_names[star2_name]
                
                coord1 = (star1['ra'], star1['dec'])
                coord2 = (star2['ra'], star2['dec'])
                
                pixel1 = self._coord_to_pixel(coord1, bounds, self.image_size)
                pixel2 = self._coord_to_pixel(coord2, bounds, self.image_size)
                
                cv2.line(image, pixel1, pixel2, self.constellation_line_color, 2)
        
        # Draw stars
        for star in stars:
            coord = (star['ra'], star['dec'])
            pixel = self._coord_to_pixel(coord, bounds, self.image_size)
            
            # Get star properties
            color = self._get_star_color(star['spectral'])
            size = self._get_star_size(star['mag'])
            
            # Draw star
            cv2.circle(image, pixel, size, color, -1)
            
            # Add white center for bright stars
            if star['mag'] <= 2:
                cv2.circle(image, pixel, max(1, size//2), (255, 255, 255), -1)
        
        # Convert to PIL for text
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        
        # Try to load font
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Add star labels
        for star in stars:
            coord = (star['ra'], star['dec'])
            pixel = self._coord_to_pixel(coord, bounds, self.image_size)
            
            # Create label
            label = f"{star['bayer']} {star['name']}"
            if star['hr']:
                label += f" (HR{star['hr']})"
            
            # Draw label
            draw.text((pixel[0] + 10, pixel[1] - 10), label, fill=self.text_color, font=font)
        
        # Add constellation title
        title = f"{constellation_name} ({const_data['iau_code']})"
        subtitle = f"{const_data['description']} - {const_data['hemisphere'].title()} Hemisphere"
        
        draw.text((20, 20), title, fill=self.text_color, font=font)
        draw.text((20, 40), subtitle, fill=self.text_color, font=font)
        
        # Add professional info
        info_text = f"Stars: {len(stars)} | Messier Objects: {len(const_data.get('messier_objects', []))}"
        draw.text((20, self.image_size[1] - 40), info_text, fill=self.text_color, font=font)
        
        # Add Yale BSC reference
        bsc_text = "Based on Yale Bright Star Catalogue"
        draw.text((20, self.image_size[1] - 20), bsc_text, fill=self.text_color, font=font)
        
        return np.array(pil_image)
    
    def create_all_constellation_images(self):
        """Create images for all constellations."""
        print("🌟 Creating professional constellation images...")
        print(f"   Based on Yale Bright Star Catalogue standards")
        print(f"   Total constellations: {len(self.constellations)}")
        
        for const_name in self.constellations:
            print(f"   Creating {const_name}...")
            image = self.create_constellation_image(const_name)
            
            if image is not None:
                output_path = f"Output/professional_{const_name.lower()}.png"
                cv2.imwrite(output_path, image)
                print(f"     ✅ Saved to {output_path}")
        
        print(f"✅ Created {len(self.constellations)} professional constellation images")
    
    def create_professional_atlas(self) -> np.ndarray:
        """Create a professional constellation atlas."""
        print("🗺️ Creating professional constellation atlas...")
        
        # Calculate grid size
        n_constellations = len(self.constellations)
        grid_size = int(np.ceil(np.sqrt(n_constellations)))
        
        # Create atlas image
        atlas_width = grid_size * self.image_size[0]
        atlas_height = grid_size * self.image_size[1]
        atlas = np.full((atlas_height, atlas_width, 3), self.background_color, dtype=np.uint8)
        
        # Add constellations to atlas
        const_names = list(self.constellations.keys())
        for i, const_name in enumerate(const_names):
            row = i // grid_size
            col = i % grid_size
            
            image = self.create_constellation_image(const_name)
            if image is not None:
                y_start = row * self.image_size[1]
                y_end = y_start + self.image_size[1]
                x_start = col * self.image_size[0]
                x_end = x_start + self.image_size[0]
                
                atlas[y_start:y_end, x_start:x_end] = image
        
        # Add atlas title
        pil_atlas = Image.fromarray(atlas)
        draw = ImageDraw.Draw(pil_atlas)
        
        try:
            title_font = ImageFont.truetype("arial.ttf", 32)
            subtitle_font = ImageFont.truetype("arial.ttf", 20)
        except:
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
        
        # Add title
        title = "Professional Constellation Atlas"
        subtitle = f"Based on Yale Bright Star Catalogue - {len(self.constellations)} Constellations"
        
        draw.text((20, 20), title, fill=self.text_color, font=title_font)
        draw.text((20, 60), subtitle, fill=self.text_color, font=subtitle_font)
        
        # Add reference
        ref_text = "Martin Krzywinski Database + Yale BSC Standards"
        draw.text((20, atlas_height - 40), ref_text, fill=self.text_color, font=subtitle_font)
        
        return np.array(pil_atlas)

def main():
    """Create professional constellation visualizations."""
    print("🌟 Professional Constellation Visualizer")
    print("=" * 50)
    print("Based on Yale Bright Star Catalogue and Martin Krzywinski standards")
    print("=" * 50)
    
    # Create visualizer
    visualizer = ProfessionalConstellationVisualizer()
    
    if not visualizer.constellations:
        print("❌ No constellation data available")
        return
    
    # Create individual constellation images
    visualizer.create_all_constellation_images()
    
    # Create professional atlas
    print(f"\n🗺️ Creating professional atlas...")
    atlas = visualizer.create_professional_atlas()
    
    if atlas is not None:
        atlas_path = "Output/professional_constellation_atlas.png"
        cv2.imwrite(atlas_path, atlas)
        print(f"✅ Professional atlas saved to {atlas_path}")
    
    print(f"\n📁 Professional constellation visualizations complete!")
    print(f"   Individual images: Output/professional_*.png")
    print(f"   Atlas: Output/professional_constellation_atlas.png")
    print(f"   Based on Yale Bright Star Catalogue standards")

if __name__ == "__main__":
    main() 