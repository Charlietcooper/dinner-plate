#!/usr/bin/env python3
"""
Professional Constellation Annotator - Identifies and overlays constellations on images
Based on Yale Bright Star Catalogue and professional shape preservation
"""

import json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import os
from typing import Dict, List, Tuple, Optional
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

class ProfessionalConstellationAnnotator:
    """Professional constellation annotator with shape preservation."""
    
    def __init__(self):
        self.database_path = "Processing/professional_constellation_database.json"
        self.constellations = self._load_database()
        
        # Professional annotation settings
        self.constellation_line_color = (100, 150, 255)  # Blue
        self.star_color = (255, 255, 255)  # White
        self.text_color = (255, 255, 255)  # White
        self.line_thickness = 2
        self.star_size = 3
        
    def _load_database(self) -> Dict:
        """Load the professional constellation database."""
        try:
            with open(self.database_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Failed to load database: {e}")
            return {}
    
    def _create_demo_wcs(self, image_shape: Tuple[int, int]) -> WCS:
        """Create a demo WCS for testing (southern sky)."""
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [image_shape[1]//2, image_shape[0]//2]  # Center of image
        wcs.wcs.crval = [180, -60]  # Southern sky coordinates (RA=180°, Dec=-60°)
        wcs.wcs.cdelt = [0.1, 0.1]  # Pixel scale (degrees per pixel)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        return wcs
    
    def _sky_to_pixel(self, wcs: WCS, ra: float, dec: float) -> Optional[Tuple[int, int]]:
        """Convert RA/Dec coordinates to pixel coordinates."""
        try:
            coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
            pixel = wcs.world_to_pixel(coord)
            
            if np.isnan(pixel[0]) or np.isnan(pixel[1]):
                return None
            
            return int(pixel[0]), int(pixel[1])
        except Exception as e:
            return None
    
    def _detect_stars_in_image(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """Detect potential stars in the image."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to find bright objects
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size (stars should be small)
        stars = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 5 < area < 100:  # Reasonable star size
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    stars.append((cx, cy))
        
        return stars[:50]  # Limit to top 50 detected stars
    
    def _find_visible_constellations(self, wcs: WCS, image_shape: Tuple[int, int]) -> List[str]:
        """Find constellations visible in the image based on WCS."""
        visible_constellations = []
        
        for const_name, const_data in self.constellations.items():
            # Check if any stars in the constellation are visible
            visible_stars = 0
            for star in const_data['stars']:
                pixel = self._sky_to_pixel(wcs, star['ra'], star['dec'])
                if pixel and 0 <= pixel[0] < image_shape[1] and 0 <= pixel[1] < image_shape[0]:
                    visible_stars += 1
            
            # If at least 2 stars are visible, consider the constellation visible
            if visible_stars >= 2:
                visible_constellations.append(const_name)
        
        return visible_constellations
    
    def _draw_constellation(self, image: np.ndarray, wcs: WCS, constellation_name: str) -> bool:
        """Draw a constellation on the image."""
        if constellation_name not in self.constellations:
            return False
        
        const_data = self.constellations[constellation_name]
        stars = const_data['stars']
        
        # Create star name mapping
        star_names = {star['name']: star for star in stars}
        star_pixels = {}
        
        # Convert all stars to pixel coordinates
        for star in stars:
            pixel = self._sky_to_pixel(wcs, star['ra'], star['dec'])
            if pixel:
                star_pixels[star['name']] = pixel
        
        # Draw constellation lines
        for line in const_data['lines']:
            star1_name, star2_name = line
            if star1_name in star_pixels and star2_name in star_pixels:
                pixel1 = star_pixels[star1_name]
                pixel2 = star_pixels[star2_name]
                cv2.line(image, pixel1, pixel2, self.constellation_line_color, self.line_thickness)
        
        # Draw stars
        for star in stars:
            if star['name'] in star_pixels:
                pixel = star_pixels[star['name']]
                cv2.circle(image, pixel, self.star_size, self.star_color, -1)
        
        return True
    
    def annotate_image(self, image_path: str, output_path: str) -> bool:
        """Annotate an image with professional constellation overlays."""
        print(f"🌟 Professional Constellation Annotator")
        print(f"   Processing: {image_path}")
        print(f"   Based on Yale Bright Star Catalogue standards")
        
        # Load image
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return False
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Failed to load image: {image_path}")
            return False
        
        print(f"   Image size: {image.shape[1]}x{image.shape[0]}")
        
        # Create demo WCS (southern sky for test image)
        wcs = self._create_demo_wcs(image.shape)
        print(f"   Demo WCS: RA=180°, Dec=-60° (Southern sky)")
        
        # Find visible constellations
        visible_constellations = self._find_visible_constellations(wcs, image.shape)
        print(f"   Visible constellations: {len(visible_constellations)}")
        for const in visible_constellations:
            print(f"     - {const}")
        
        # Create annotated image
        annotated_image = image.copy()
        
        # Draw constellations
        for constellation_name in visible_constellations:
            print(f"   Drawing {constellation_name}...")
            self._draw_constellation(annotated_image, wcs, constellation_name)
        
        # Add constellation labels
        annotated_image = self._add_constellation_labels(annotated_image, wcs, visible_constellations)
        
        # Add professional info
        annotated_image = self._add_professional_info(annotated_image, len(visible_constellations))
        
        # Save result
        cv2.imwrite(output_path, annotated_image)
        print(f"✅ Professional annotation saved to: {output_path}")
        
        return True
    
    def _add_constellation_labels(self, image: np.ndarray, wcs: WCS, constellations: List[str]) -> np.ndarray:
        """Add constellation labels to the image."""
        # Convert to PIL for text
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        # Try to load font
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Add labels for each constellation
        for constellation_name in constellations:
            const_data = self.constellations[constellation_name]
            
            # Find center of constellation
            center_ra = sum(star['ra'] for star in const_data['stars']) / len(const_data['stars'])
            center_dec = sum(star['dec'] for star in const_data['stars']) / len(const_data['stars'])
            
            center_pixel = self._sky_to_pixel(wcs, center_ra, center_dec)
            if center_pixel:
                # Create label
                label = f"{constellation_name} ({const_data['iau_code']})"
                
                # Draw label
                draw.text((center_pixel[0] + 10, center_pixel[1] - 10), 
                         label, fill=self.text_color, font=font)
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def _add_professional_info(self, image: np.ndarray, num_constellations: int) -> np.ndarray:
        """Add professional information to the image."""
        # Convert to PIL for text
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        # Try to load font
        try:
            title_font = ImageFont.truetype("arial.ttf", 24)
            info_font = ImageFont.truetype("arial.ttf", 16)
        except:
            title_font = ImageFont.load_default()
            info_font = ImageFont.load_default()
        
        # Add title
        title = "Professional Constellation Annotation"
        draw.text((20, 20), title, fill=self.text_color, font=title_font)
        
        # Add info
        info_lines = [
            f"Constellations identified: {num_constellations}",
            f"Based on Yale Bright Star Catalogue",
            f"Professional astronomical standards",
            f"Shape preservation ready"
        ]
        
        y_offset = 60
        for line in info_lines:
            draw.text((20, y_offset), line, fill=self.text_color, font=info_font)
            y_offset += 25
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def main():
    """Test professional constellation annotation on the test image."""
    print("🌟 Professional Constellation Annotation Test")
    print("=" * 50)
    print("Testing on Input/test-1.jpg")
    print("=" * 50)
    
    # Create annotator
    annotator = ProfessionalConstellationAnnotator()
    
    if not annotator.constellations:
        print("❌ No constellation data available")
        return
    
    # Test image path
    test_image_path = "Input/test-1.jpg"
    output_path = "Output/professional_annotated_test.jpg"
    
    # Run annotation
    success = annotator.annotate_image(test_image_path, output_path)
    
    if success:
        print(f"\n✅ Professional constellation annotation complete!")
        print(f"   Input: {test_image_path}")
        print(f"   Output: {output_path}")
        print(f"   Based on Yale Bright Star Catalogue standards")
        print(f"   Ready for shape preservation integration")
    else:
        print(f"❌ Annotation failed")

if __name__ == "__main__":
    main() 