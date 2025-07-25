#!/usr/bin/env python3
"""
Constellation Annotator for Wide-Field Astronomical Images

This script takes a wide-field astronomical image (JPEG, PNG, FITS) and overlays
constellation lines on it using plate solving to determine the celestial coordinates.

Usage:
    python constellation_annotator.py input_image.jpg output_image.jpg
    python constellation_annotator.py input_image.jpg output_image.jpg --api-key YOUR_API_KEY

Requirements:
    - Astroquery (for plate solving via Astrometry.net)
    - Astropy (for coordinate transformations)
    - OpenCV (for image processing and drawing)
    - NumPy (for array operations)
    - Pillow (for image I/O)
    - Requests (for API calls)

Author: AI Assistant
Date: 2024
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import requests

# Astronomy libraries
try:
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    from astroquery.astrometry_net import AstrometryNet
    from astroquery.vizier import Vizier
    from astroquery.simbad import Simbad
except ImportError as e:
    print(f"Error: Missing astronomy library: {e}")
    print("Please install required packages:")
    print("pip install astropy astroquery")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from constellation_data_enhanced import EnhancedConstellationData
from camera_config import get_camera_config, get_plate_solve_params

class ConstellationData:
    """Manages constellation line data and star information."""
    
    def __init__(self):
        # Use the enhanced constellation data
        self.enhanced_data = EnhancedConstellationData()
        self.constellation_lines = self.enhanced_data.constellation_lines
        self.bright_stars = self.enhanced_data.bright_stars
    
    def _load_constellation_lines(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        Load constellation line definitions.
        Returns a dictionary mapping constellation names to lists of star pairs.
        """
        # This is a simplified version - in practice, you'd load from a comprehensive database
        # For now, we'll include some major constellations with their brightest stars
        return {
            'Orion': [
                ('Betelgeuse', 'Bellatrix'),
                ('Bellatrix', 'Mintaka'),
                ('Mintaka', 'Alnilam'),
                ('Alnilam', 'Alnitak'),
                ('Alnitak', 'Saiph'),
                ('Saiph', 'Rigel'),
                ('Rigel', 'Betelgeuse'),
                ('Mintaka', 'Alnitak'),  # Belt
                ('Alnilam', 'Saiph'),    # Belt
            ],
            'Ursa Major': [
                ('Dubhe', 'Merak'),
                ('Merak', 'Phecda'),
                ('Phecda', 'Megrez'),
                ('Megrez', 'Alioth'),
                ('Alioth', 'Mizar'),
                ('Mizar', 'Alkaid'),
                ('Dubhe', 'Phecda'),
                ('Merak', 'Megrez'),
            ],
            'Cassiopeia': [
                ('Schedar', 'Caph'),
                ('Caph', 'Cih'),
                ('Cih', 'Ruchbah'),
                ('Ruchbah', 'Segin'),
                ('Segin', 'Schedar'),
            ],
            'Cygnus': [
                ('Deneb', 'Sadr'),
                ('Sadr', 'Gienah'),
                ('Gienah', 'Delta Cygni'),
                ('Delta Cygni', 'Albireo'),
                ('Albireo', 'Sadr'),
            ],
            'Scorpius': [
                ('Antares', 'Shaula'),
                ('Shaula', 'Lesath'),
                ('Lesath', 'Dschubba'),
                ('Dschubba', 'Antares'),
                ('Antares', 'Acrab'),
                ('Acrab', 'Dschubba'),
            ],
            'Sagittarius': [
                ('Kaus Australis', 'Kaus Media'),
                ('Kaus Media', 'Kaus Borealis'),
                ('Kaus Borealis', 'Nunki'),
                ('Nunki', 'Ascella'),
                ('Ascella', 'Kaus Australis'),
            ]
        }
    
    def _load_bright_stars(self) -> Dict[str, Tuple[float, float]]:
        """
        Load bright star coordinates (RA, Dec in degrees).
        Returns a dictionary mapping star names to (RA, Dec) tuples.
        """
        # This is a simplified version - in practice, you'd load from a comprehensive catalog
        return {
            # Orion
            'Betelgeuse': (88.7929, 7.4071),
            'Bellatrix': (81.2828, 6.3497),
            'Mintaka': (83.0016, -0.2991),
            'Alnilam': (84.0534, -1.2019),
            'Alnitak': (85.1897, -1.9426),
            'Saiph': (86.9391, -9.6696),
            'Rigel': (78.6345, -8.2016),
            
            # Ursa Major
            'Dubhe': (165.9319, 61.7510),
            'Merak': (165.4603, 56.3824),
            'Phecda': (178.4577, 53.6948),
            'Megrez': (183.8565, 57.0326),
            'Alioth': (193.5073, 55.9598),
            'Mizar': (200.9814, 54.9254),
            'Alkaid': (206.8852, 49.3133),
            
            # Cassiopeia
            'Schedar': (10.1268, 56.5373),
            'Caph': (2.2945, 59.1498),
            'Cih': (14.1651, 60.7167),
            'Ruchbah': (21.4538, 60.2353),
            'Segin': (28.5988, 63.6701),
            
            # Cygnus
            'Deneb': (310.3580, 45.2803),
            'Sadr': (305.5571, 40.2567),
            'Gienah': (318.2341, 40.2567),
            'Delta Cygni': (308.3039, 45.1313),
            'Albireo': (292.6804, 27.9597),
            
            # Scorpius
            'Antares': (247.3519, -26.4320),
            'Shaula': (263.4022, -37.1038),
            'Lesath': (264.3297, -37.2958),
            'Dschubba': (240.0833, -22.6217),
            'Acrab': (241.3592, -19.8054),
            
            # Sagittarius
            'Kaus Australis': (276.0430, -34.3846),
            'Kaus Media': (274.4067, -29.8281),
            'Kaus Borealis': (271.4520, -25.4217),
            'Nunki': (283.8164, -26.2967),
            'Ascella': (287.4407, -29.8811),
        }
    
    def get_constellation_stars(self, constellation_name: str) -> List[Tuple[str, str]]:
        """Get the star pairs that form lines for a given constellation."""
        return self.constellation_lines.get(constellation_name, [])
    
    def get_star_coordinates(self, star_name: str) -> Optional[Tuple[float, float]]:
        """Get the RA/Dec coordinates for a given star."""
        return self.bright_stars.get(star_name)
    
    def get_all_constellations(self) -> List[str]:
        """Get list of all available constellations."""
        return list(self.constellation_lines.keys())

class PlateSolver:
    """Handles plate solving using Astrometry.net."""
    
    def __init__(self, api_key: Optional[str] = None, camera_config: str = "canon_200d"):
        self.api_key = api_key or os.getenv('ASTROMETRY_API_KEY')
        if not self.api_key:
            logger.warning("No Astrometry.net API key provided. Plate solving may fail.")
        
        self.ast = AstrometryNet()
        if self.api_key:
            self.ast.api_key = self.api_key
        
        # Get camera configuration
        self.camera_config = get_camera_config(camera_config)
        self.plate_solve_params = get_plate_solve_params(self.camera_config)
        
        logger.info(f"Using camera: {self.camera_config['camera']} with {self.camera_config['lens']}")
    
    def solve_image(self, image_path: str) -> Optional[WCS]:
        """
        Plate solve an image and return the WCS solution.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            WCS object if successful, None otherwise
        """
        try:
            logger.info(f"Starting plate solve for {image_path}")
            logger.info(f"Camera: {self.camera_config['camera']} at {self.camera_config['focal_length_mm']}mm")
            
            # Get plate solving parameters from camera config
            solve_params = {
                'solve_timeout': 120,
                'submission_id': None,
                'scale_est': self.plate_solve_params['scale_est'],
                'scale_lower': self.plate_solve_params['scale_lower'],
                'scale_upper': self.plate_solve_params['scale_upper'],
                'scale_units': self.plate_solve_params['scale_units'],
                'radius': self.plate_solve_params['radius'],
            }
            
            # Try to solve the image with camera-specific parameters
            wcs_header = self.ast.solve_from_image(image_path, **solve_params)
            
            if wcs_header:
                logger.info("Plate solve successful!")
                return WCS(wcs_header)
            else:
                logger.error("Plate solve failed - no WCS solution found")
                return None
                
        except Exception as e:
            logger.error(f"Plate solve error: {e}")
            return None

class ConstellationAnnotator:
    """Main class for annotating images with constellation lines."""
    
    def __init__(self, api_key: Optional[str] = None, camera_config: str = "canon_200d"):
        self.plate_solver = PlateSolver(api_key, camera_config)
        self.constellation_data = ConstellationData()
        
        # Drawing parameters - style matching reference image
        self.line_color = (255, 0, 0)  # Blue (BGR format)
        self.line_thickness = 1  # Thin lines
        self.star_color = (255, 255, 255)  # White stars
        self.star_radius = 2  # Small stars
        self.text_color = (255, 255, 255)  # White text
        self.font_size = 12  # Smaller font for better fit
        
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load an image file (JPEG, PNG, or FITS)."""
        try:
            if image_path.lower().endswith('.fits'):
                # Load FITS file
                with fits.open(image_path) as hdul:
                    data = hdul[0].data
                    if data is None:
                        logger.error("No data found in FITS file")
                        return None
                    
                    # Normalize FITS data to 0-255 range
                    data_min = np.min(data)
                    data_max = np.max(data)
                    if data_max > data_min:
                        data = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
                    else:
                        data = np.zeros_like(data, dtype=np.uint8)
                    
                    # Convert to RGB if needed
                    if len(data.shape) == 2:
                        data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
                    
                    return data
            else:
                # Load regular image file
                image = cv2.imread(image_path)
                if image is None:
                    logger.error(f"Could not load image: {image_path}")
                    return None
                return image
                
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def save_image(self, image: np.ndarray, output_path: str) -> bool:
        """Save an image to file."""
        try:
            success = cv2.imwrite(output_path, image)
            if success:
                logger.info(f"Image saved to {output_path}")
                return True
            else:
                logger.error(f"Failed to save image to {output_path}")
                return False
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return False
    
    def sky_to_pixel(self, wcs: WCS, ra: float, dec: float) -> Optional[Tuple[float, float]]:
        """
        Convert sky coordinates (RA, Dec) to pixel coordinates.
        
        Args:
            wcs: WCS object from plate solve
            ra: Right Ascension in degrees
            dec: Declination in degrees
            
        Returns:
            (x, y) pixel coordinates or None if conversion fails
        """
        try:
            # Create SkyCoord object
            coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
            
            # Convert to pixel coordinates
            x, y = wcs.world_to_pixel(coord)
            
            # Check if coordinates are within image bounds
            if np.isfinite(x) and np.isfinite(y):
                return (float(x), float(y))
            else:
                return None
                
        except Exception as e:
            logger.debug(f"Coordinate conversion failed for RA={ra}, Dec={dec}: {e}")
            return None
    
    def draw_constellation_lines(self, image: np.ndarray, wcs: WCS) -> np.ndarray:
        """
        Draw constellation lines on the image.
        
        Args:
            image: Input image as numpy array
            wcs: WCS object from plate solve
            
        Returns:
            Image with constellation lines drawn
        """
        # Create a copy of the image for drawing
        annotated_image = image.copy()
        
        # Track which constellations are visible
        visible_constellations = []
        
        # Draw lines for each constellation
        for constellation_name in self.constellation_data.get_all_constellations():
            constellation_visible = False
            star_pairs = self.constellation_data.get_constellation_stars(constellation_name)
            
            for star1_name, star2_name in star_pairs:
                # Get coordinates for both stars
                star1_coords = self.constellation_data.get_star_coordinates(star1_name)
                star2_coords = self.constellation_data.get_star_coordinates(star2_name)
                
                if star1_coords and star2_coords:
                    # Convert to pixel coordinates
                    pixel1 = self.sky_to_pixel(wcs, star1_coords[0], star1_coords[1])
                    pixel2 = self.sky_to_pixel(wcs, star2_coords[0], star2_coords[1])
                    
                    if pixel1 and pixel2:
                        # Check if both points are within image bounds
                        height, width = image.shape[:2]
                        if (0 <= pixel1[0] < width and 0 <= pixel1[1] < height and
                            0 <= pixel2[0] < width and 0 <= pixel2[1] < height):
                            
                            # Draw the line (thin blue line)
                            cv2.line(annotated_image, 
                                   (int(pixel1[0]), int(pixel1[1])),
                                   (int(pixel2[0]), int(pixel2[1])),
                                   self.line_color, self.line_thickness)
                            
                            # Mark the stars with small white dots
                            cv2.circle(annotated_image, 
                                     (int(pixel1[0]), int(pixel1[1])),
                                     self.star_radius, self.star_color, -1)
                            cv2.circle(annotated_image, 
                                     (int(pixel2[0]), int(pixel2[1])),
                                     self.star_radius, self.star_color, -1)
                            
                            constellation_visible = True
            
            if constellation_visible:
                visible_constellations.append(constellation_name)
        
        # Add constellation labels
        if visible_constellations:
            self._add_constellation_labels(annotated_image, wcs, visible_constellations)
        
        logger.info(f"Visible constellations: {', '.join(visible_constellations)}")
        return annotated_image
    
    def _add_constellation_labels(self, image: np.ndarray, wcs: WCS, 
                                constellation_names: List[str]) -> None:
        """Add text labels for visible constellations."""
        try:
            # Convert to PIL for better text handling
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            
            # Try to load a font
            try:
                font = ImageFont.truetype("arial.ttf", self.font_size)
            except:
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", self.font_size)
                except:
                    font = ImageFont.load_default()
            
            # Track label positions to avoid overlap
            label_positions = []
            
            for constellation_name in constellation_names:
                # Find a representative star for this constellation
                star_pairs = self.constellation_data.get_constellation_stars(constellation_name)
                if star_pairs:
                    # Use the first star in the first pair
                    star_name = star_pairs[0][0]
                    star_coords = self.constellation_data.get_star_coordinates(star_name)
                    
                    if star_coords:
                        pixel = self.sky_to_pixel(wcs, star_coords[0], star_coords[1])
                        if pixel:
                            # Calculate label position (center of constellation)
                            x, y = int(pixel[0]), int(pixel[1])
                            
                            # Check for overlap with existing labels
                            overlap = False
                            for pos in label_positions:
                                if abs(x - pos[0]) < 50 and abs(y - pos[1]) < 20:
                                    overlap = True
                                    break
                            
                            if not overlap:
                                # Draw text with slight offset
                                text_x, text_y = x + 5, y - 5
                                draw.text((text_x, text_y), constellation_name, 
                                        fill=self.text_color, font=font)
                                label_positions.append((x, y))
            
            # Convert back to OpenCV format
            image[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            logger.warning(f"Could not add constellation labels: {e}")
    
    def annotate_image(self, input_path: str, output_path: str) -> bool:
        """
        Main method to annotate an image with constellation lines.
        
        Args:
            input_path: Path to input image
            output_path: Path for output annotated image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load the image
            logger.info(f"Loading image: {input_path}")
            image = self.load_image(input_path)
            if image is None:
                return False
            
            # Plate solve the image
            logger.info("Starting plate solve...")
            wcs = self.plate_solver.solve_image(input_path)
            if wcs is None:
                logger.error("Plate solving failed. Cannot proceed with annotation.")
                return False
            
            # Draw constellation lines
            logger.info("Drawing constellation lines...")
            annotated_image = self.draw_constellation_lines(image, wcs)
            
            # Save the result
            return self.save_image(annotated_image, output_path)
            
        except Exception as e:
            logger.error(f"Error during annotation: {e}")
            return False

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Annotate wide-field astronomical images with constellation lines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python constellation_annotator.py input.jpg output.jpg
  python constellation_annotator.py input.fits output.jpg --api-key YOUR_KEY
  
Note: You need an Astrometry.net API key for plate solving.
Get one at: https://nova.astrometry.net/
        """
    )
    
    parser.add_argument('input', help='Input image file (JPEG, PNG, or FITS)')
    parser.add_argument('output', help='Output image file')
    parser.add_argument('--api-key', help='Astrometry.net API key (or set ASTROMETRY_API_KEY env var)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check input file
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create annotator and process image
    annotator = ConstellationAnnotator(args.api_key)
    
    logger.info("Starting constellation annotation...")
    success = annotator.annotate_image(args.input, args.output)
    
    if success:
        logger.info("Annotation completed successfully!")
        sys.exit(0)
    else:
        logger.error("Annotation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 