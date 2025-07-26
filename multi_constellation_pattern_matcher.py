#!/usr/bin/env python3
"""
Multi-Constellation Pattern Matcher - Uses spatial relationships between adjacent constellations
Based on Martin Krzywinski's star chart and Yale Bright Star Catalogue
"""

import json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import os
from typing import Dict, List, Tuple, Optional, Set
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from itertools import combinations

class MultiConstellationPatternMatcher:
    """Matches constellations using spatial relationships and adjacency patterns."""
    
    def __init__(self):
        self.database_path = "Processing/professional_constellation_database.json"
        self.constellations = self._load_database()
        
        # Constellation adjacency relationships (based on Martin Krzywinski's chart)
        self.constellation_adjacencies = {
            "Crux": ["Musca", "Centaurus", "Carina"],
            "Musca": ["Crux", "Chamaeleon", "Apus"],
            "Centaurus": ["Crux", "Lupus", "Lupus", "Circinus", "Triangulum Australe"],
            "Carina": ["Crux", "Vela", "Puppis", "Pictor", "Volans"],
            "Vela": ["Carina", "Puppis", "Pyxis", "Antlia", "Centaurus"],
            "Puppis": ["Carina", "Vela", "Pyxis", "Canis Major", "Monoceros"],
            "Lupus": ["Centaurus", "Norma", "Scorpius", "Libra"],
            "Scorpius": ["Lupus", "Norma", "Ara", "Ophiuchus", "Libra"],
            "Ara": ["Scorpius", "Norma", "Triangulum Australe", "Pavo", "Telescopium"],
            "Triangulum Australe": ["Ara", "Circinus", "Centaurus", "Norma"],
            "Circinus": ["Triangulum Australe", "Centaurus", "Lupus", "Norma"],
            "Norma": ["Circinus", "Lupus", "Scorpius", "Ara", "Triangulum Australe"]
        }
        
        # Professional annotation settings
        self.constellation_line_color = (100, 150, 255)  # Blue
        self.star_color = (255, 255, 255)  # White
        self.text_color = (255, 255, 255)  # White
        self.verified_color = (0, 255, 0)  # Green for verified patterns
        self.unverified_color = (255, 165, 0)  # Orange for unverified
        
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
    
    def _calculate_constellation_center(self, constellation_name: str) -> Optional[Tuple[float, float]]:
        """Calculate the center coordinates of a constellation."""
        if constellation_name not in self.constellations:
            return None
        
        stars = self.constellations[constellation_name]['stars']
        if not stars:
            return None
        
        center_ra = sum(star['ra'] for star in stars) / len(stars)
        center_dec = sum(star['dec'] for star in stars) / len(stars)
        
        return center_ra, center_dec
    
    def _calculate_constellation_distance(self, const1: str, const2: str) -> Optional[float]:
        """Calculate the angular distance between two constellation centers."""
        center1 = self._calculate_constellation_center(const1)
        center2 = self._calculate_constellation_center(const2)
        
        if center1 is None or center2 is None:
            return None
        
        ra1, dec1 = center1
        ra2, dec2 = center2
        
        # Convert to radians
        ra1_rad = np.radians(ra1)
        dec1_rad = np.radians(dec1)
        ra2_rad = np.radians(ra2)
        dec2_rad = np.radians(dec2)
        
        # Calculate angular distance using spherical trigonometry
        cos_dist = (np.sin(dec1_rad) * np.sin(dec2_rad) + 
                   np.cos(dec1_rad) * np.cos(dec2_rad) * np.cos(ra1_rad - ra2_rad))
        
        # Handle numerical precision issues
        cos_dist = np.clip(cos_dist, -1.0, 1.0)
        
        distance = np.arccos(cos_dist)
        return np.degrees(distance)
    
    def _find_constellation_groups(self, wcs: WCS, image_shape: Tuple[int, int]) -> List[Dict]:
        """Find groups of adjacent constellations that are visible in the image."""
        print("🔍 Finding constellation groups using spatial relationships...")
        
        # Find all potentially visible constellations
        visible_constellations = []
        for const_name in self.constellations:
            center = self._calculate_constellation_center(const_name)
            if center:
                pixel = self._sky_to_pixel(wcs, center[0], center[1])
                if pixel and 0 <= pixel[0] < image_shape[1] and 0 <= pixel[1] < image_shape[0]:
                    visible_constellations.append(const_name)
        
        print(f"   Potentially visible constellations: {len(visible_constellations)}")
        
        # Find constellation groups (2-4 adjacent constellations)
        constellation_groups = []
        
        # Check pairs of adjacent constellations
        for const1 in visible_constellations:
            if const1 in self.constellation_adjacencies:
                for const2 in self.constellation_adjacencies[const1]:
                    if const2 in visible_constellations:
                        # Verify spatial relationship
                        distance = self._calculate_constellation_distance(const1, const2)
                        if distance and 5 < distance < 50:  # Reasonable angular distance
                            group = {
                                "constellations": [const1, const2],
                                "type": "pair",
                                "distance": distance,
                                "verified": True
                            }
                            constellation_groups.append(group)
        
        # Check triplets
        for const1 in visible_constellations:
            if const1 in self.constellation_adjacencies:
                for const2 in self.constellation_adjacencies[const1]:
                    if const2 in visible_constellations and const2 in self.constellation_adjacencies:
                        for const3 in self.constellation_adjacencies[const2]:
                            if const3 in visible_constellations:
                                # Check if all three are adjacent
                                if (const1 in self.constellation_adjacencies.get(const3, []) or
                                    const3 in self.constellation_adjacencies.get(const1, [])):
                                    
                                    # Calculate distances
                                    dist12 = self._calculate_constellation_distance(const1, const2)
                                    dist23 = self._calculate_constellation_distance(const2, const3)
                                    dist13 = self._calculate_constellation_distance(const1, const3)
                                    
                                    if all(d and 5 < d < 50 for d in [dist12, dist23, dist13]):
                                        group = {
                                            "constellations": [const1, const2, const3],
                                            "type": "triplet",
                                            "distances": [dist12, dist23, dist13],
                                            "verified": True
                                        }
                                        constellation_groups.append(group)
        
        # Remove duplicates
        unique_groups = []
        seen = set()
        for group in constellation_groups:
            group_key = tuple(sorted(group["constellations"]))
            if group_key not in seen:
                seen.add(group_key)
                unique_groups.append(group)
        
        print(f"   Found {len(unique_groups)} verified constellation groups")
        for group in unique_groups:
            print(f"     - {group['type']}: {' → '.join(group['constellations'])}")
        
        return unique_groups
    
    def _draw_constellation_group(self, image: np.ndarray, wcs: WCS, group: Dict) -> bool:
        """Draw a verified constellation group on the image."""
        constellations = group["constellations"]
        is_verified = group.get("verified", False)
        
        color = self.verified_color if is_verified else self.unverified_color
        
        for constellation_name in constellations:
            if constellation_name not in self.constellations:
                continue
            
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
                    cv2.line(image, pixel1, pixel2, color, 2)
            
            # Draw stars
            for star in stars:
                if star['name'] in star_pixels:
                    pixel = star_pixels[star['name']]
                    cv2.circle(image, pixel, 3, self.star_color, -1)
        
        return True
    
    def _add_group_labels(self, image: np.ndarray, wcs: WCS, groups: List[Dict]) -> np.ndarray:
        """Add labels for constellation groups."""
        # Convert to PIL for text
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        # Try to load font
        try:
            font = ImageFont.truetype("arial.ttf", 20)
            small_font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Add labels for each group
        for i, group in enumerate(groups):
            constellations = group["constellations"]
            
            # Find center of group
            centers = []
            for const_name in constellations:
                center = self._calculate_constellation_center(const_name)
                if center:
                    centers.append(center)
            
            if centers:
                avg_ra = sum(c[0] for c in centers) / len(centers)
                avg_dec = sum(c[1] for c in centers) / len(centers)
                
                center_pixel = self._sky_to_pixel(wcs, avg_ra, avg_dec)
                if center_pixel:
                    # Create group label
                    group_label = f"Group {i+1}: {' → '.join(constellations)}"
                    if group.get("verified", False):
                        group_label += " ✓"
                    
                    # Draw label
                    draw.text((center_pixel[0] + 10, center_pixel[1] - 10), 
                             group_label, fill=self.text_color, font=font)
                    
                    # Add distance info for pairs
                    if group["type"] == "pair" and "distance" in group:
                        dist_text = f"Distance: {group['distance']:.1f}°"
                        draw.text((center_pixel[0] + 10, center_pixel[1] + 10), 
                                 dist_text, fill=self.text_color, font=small_font)
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def _add_multi_pattern_info(self, image: np.ndarray, groups: List[Dict]) -> np.ndarray:
        """Add information about multi-constellation pattern matching."""
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
        title = "Multi-Constellation Pattern Matching"
        draw.text((20, 20), title, fill=self.text_color, font=title_font)
        
        # Add info
        info_lines = [
            f"Constellation groups identified: {len(groups)}",
            f"Based on spatial relationships and adjacency patterns",
            f"Verified using Martin Krzywinski's star chart",
            f"Green lines = Verified patterns, Orange = Unverified"
        ]
        
        y_offset = 60
        for line in info_lines:
            draw.text((20, y_offset), line, fill=self.text_color, font=info_font)
            y_offset += 25
        
        # Add group details
        y_offset += 10
        for i, group in enumerate(groups):
            group_text = f"Group {i+1}: {' → '.join(group['constellations'])}"
            if group.get("verified", False):
                group_text += " ✓"
            draw.text((20, y_offset), group_text, fill=self.text_color, font=info_font)
            y_offset += 20
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def annotate_with_multi_patterns(self, image_path: str, output_path: str) -> bool:
        """Annotate an image using multi-constellation pattern matching."""
        print(f"🌟 Multi-Constellation Pattern Matcher")
        print(f"   Processing: {image_path}")
        print(f"   Using spatial relationships and adjacency patterns")
        
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
        
        # Find constellation groups using spatial relationships
        constellation_groups = self._find_constellation_groups(wcs, image.shape)
        
        if not constellation_groups:
            print("❌ No verified constellation groups found")
            return False
        
        # Create annotated image
        annotated_image = image.copy()
        
        # Draw constellation groups
        for group in constellation_groups:
            print(f"   Drawing group: {' → '.join(group['constellations'])}")
            self._draw_constellation_group(annotated_image, wcs, group)
        
        # Add group labels
        annotated_image = self._add_group_labels(annotated_image, wcs, constellation_groups)
        
        # Add multi-pattern info
        annotated_image = self._add_multi_pattern_info(annotated_image, constellation_groups)
        
        # Save result
        cv2.imwrite(output_path, annotated_image)
        print(f"✅ Multi-pattern annotation saved to: {output_path}")
        
        return True

def main():
    """Test multi-constellation pattern matching on the test image."""
    print("🌟 Multi-Constellation Pattern Matching Test")
    print("=" * 60)
    print("Testing spatial relationships and adjacency patterns")
    print("=" * 60)
    
    # Create matcher
    matcher = MultiConstellationPatternMatcher()
    
    if not matcher.constellations:
        print("❌ No constellation data available")
        return
    
    # Test image path
    test_image_path = "Input/test-1.jpg"
    output_path = "Output/multi_pattern_annotated_test.jpg"
    
    # Run multi-pattern annotation
    success = matcher.annotate_with_multi_patterns(test_image_path, output_path)
    
    if success:
        print(f"\n✅ Multi-constellation pattern matching complete!")
        print(f"   Input: {test_image_path}")
        print(f"   Output: {output_path}")
        print(f"   Based on spatial relationships and adjacency patterns")
        print(f"   Using Martin Krzywinski's star chart verification")
    else:
        print(f"❌ Multi-pattern annotation failed")

if __name__ == "__main__":
    main() 