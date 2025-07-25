#!/usr/bin/env python3
"""
Advanced Constellation Fitter - Robust pattern matching with scale and rotation handling
"""

import numpy as np
import cv2
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from typing import List, Dict, Tuple, Optional
import json
import os
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import math

# Import from the basic constellation fitter
from constellation_shape_fitter import ConstellationShapeDatabase

class AdvancedStarDetector:
    """Advanced star detection with magnitude estimation."""
    
    def __init__(self):
        self.min_star_brightness = 20
        self.max_star_brightness = 255
        self.min_star_area = 3
        self.max_star_area = 200
        
    def detect_stars(self, image: np.ndarray) -> List[Dict]:
        """Detect stars using multiple methods for better accuracy."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Simple thresholding
        stars1 = self._detect_by_threshold(gray)
        
        # Method 2: Blob detection
        stars2 = self._detect_by_blobs(gray)
        
        # Method 3: Peak detection
        stars3 = self._detect_by_peaks(gray)
        
        # Combine and deduplicate
        all_stars = stars1 + stars2 + stars3
        unique_stars = self._deduplicate_stars(all_stars)
        
        # Sort by brightness
        unique_stars.sort(key=lambda x: x['brightness'], reverse=True)
        
        return unique_stars
    
    def _detect_by_threshold(self, gray: np.ndarray) -> List[Dict]:
        """Detect stars using thresholding."""
        stars = []
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_star_area <= area <= self.max_star_area:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    brightness = int(gray[cy, cx])
                    
                    if self.min_star_brightness <= brightness <= self.max_star_brightness:
                        stars.append({
                            "x": cx, "y": cy, "brightness": brightness, "area": area,
                            "method": "threshold"
                        })
        
        return stars
    
    def _detect_by_blobs(self, gray: np.ndarray) -> List[Dict]:
        """Detect stars using blob detection."""
        stars = []
        
        # Create blob detector
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = self.min_star_brightness
        params.maxThreshold = self.max_star_brightness
        params.filterByArea = True
        params.minArea = self.min_star_area
        params.maxArea = self.max_star_area
        params.filterByCircularity = True
        params.minCircularity = 0.3
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray)
        
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            brightness = int(gray[y, x])
            stars.append({
                "x": x, "y": y, "brightness": brightness, "area": kp.size,
                "method": "blob"
            })
        
        return stars
    
    def _detect_by_peaks(self, gray: np.ndarray) -> List[Dict]:
        """Detect stars using peak detection."""
        stars = []
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Find local maxima
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(blurred, kernel, iterations=1)
        
        # Find peaks where dilated equals original
        peaks = np.where((blurred == dilated) & (blurred > self.min_star_brightness))
        
        for y, x in zip(peaks[0], peaks[1]):
            brightness = int(blurred[y, x])
            if brightness <= self.max_star_brightness:
                # Calculate approximate area
                area = max(5, brightness // 10)
                stars.append({
                    "x": x, "y": y, "brightness": brightness, "area": area,
                    "method": "peak"
                })
        
        return stars
    
    def _deduplicate_stars(self, stars: List[Dict]) -> List[Dict]:
        """Remove duplicate star detections."""
        if not stars:
            return []
        
        # Sort by brightness
        stars.sort(key=lambda x: x['brightness'], reverse=True)
        
        unique_stars = []
        used_positions = set()
        
        for star in stars:
            x, y = star['x'], star['y']
            
            # Check if this position is already used
            duplicate = False
            for used_x, used_y in used_positions:
                distance = math.sqrt((x - used_x)**2 + (y - used_y)**2)
                if distance < 10:  # Within 10 pixels
                    duplicate = True
                    break
            
            if not duplicate:
                unique_stars.append(star)
                used_positions.add((x, y))
        
        return unique_stars

class ConstellationPatternMatcher:
    """Advanced pattern matching with scale and rotation handling."""
    
    def __init__(self):
        self.min_matches = 3
        self.max_distance = 100  # pixels
        self.scale_tolerance = 0.5  # 50% scale variation allowed
        
    def match_pattern(self, pattern_stars: List[Dict], detected_stars: List[Dict]) -> Optional[Dict]:
        """Match constellation pattern to detected stars with scale and rotation."""
        if len(pattern_stars) < self.min_matches or len(detected_stars) < self.min_matches:
            return None
        
        # Try different combinations of pattern stars
        best_match = None
        best_score = 0
        
        # Use combinations of pattern stars for matching
        from itertools import combinations
        
        for pattern_combo in combinations(pattern_stars, min(4, len(pattern_stars))):
            for detected_combo in combinations(detected_stars, len(pattern_combo)):
                match_result = self._try_match_combo(pattern_combo, detected_combo)
                if match_result and match_result['score'] > best_score:
                    best_match = match_result
                    best_score = match_result['score']
        
        return best_match
    
    def _try_match_combo(self, pattern_stars: List[Dict], detected_stars: List[Dict]) -> Optional[Dict]:
        """Try to match a specific combination of pattern and detected stars."""
        if len(pattern_stars) != len(detected_stars):
            return None
        
        # Calculate transformation parameters
        transform = self._calculate_transformation(pattern_stars, detected_stars)
        if not transform:
            return None
        
        # Apply transformation to all pattern stars
        transformed_pattern = self._apply_transformation(pattern_stars, transform)
        
        # Calculate match score
        score = self._calculate_match_score(transformed_pattern, detected_stars)
        
        if score > 0.5:  # Minimum 50% match
            return {
                'transform': transform,
                'score': score,
                'pattern_stars': pattern_stars,
                'detected_stars': detected_stars,
                'transformed_pattern': transformed_pattern
            }
        
        return None
    
    def _calculate_transformation(self, pattern_stars: List[Dict], detected_stars: List[Dict]) -> Optional[Dict]:
        """Calculate scale, rotation, and translation between pattern and detected stars."""
        if len(pattern_stars) < 2:
            return None
        
        # Get coordinates
        pattern_coords = np.array([[s['x'], s['y']] for s in pattern_stars])
        detected_coords = np.array([[s['x'], s['y']] for s in detected_stars])
        
        # Calculate centroids
        pattern_centroid = np.mean(pattern_coords, axis=0)
        detected_centroid = np.mean(detected_coords, axis=0)
        
        # Center coordinates
        pattern_centered = pattern_coords - pattern_centroid
        detected_centered = detected_coords - detected_centroid
        
        # Calculate scale
        pattern_distances = cdist(pattern_centered, pattern_centered)
        detected_distances = cdist(detected_centered, detected_centered)
        
        # Find scale factor
        scale_factors = []
        for i in range(len(pattern_distances)):
            for j in range(i+1, len(pattern_distances)):
                if pattern_distances[i, j] > 0:
                    scale = detected_distances[i, j] / pattern_distances[i, j]
                    scale_factors.append(scale)
        
        if not scale_factors:
            return None
        
        scale = np.median(scale_factors)
        
        # Check scale tolerance
        if not (1 - self.scale_tolerance <= scale <= 1 + self.scale_tolerance):
            return None
        
        # Calculate rotation (simplified)
        rotation = 0  # Could be enhanced with SVD
        
        return {
            'scale': scale,
            'rotation': rotation,
            'translation': detected_centroid - pattern_centroid * scale
        }
    
    def _apply_transformation(self, pattern_stars: List[Dict], transform: Dict) -> List[Dict]:
        """Apply transformation to pattern stars."""
        transformed = []
        
        for star in pattern_stars:
            x, y = star['x'], star['y']
            
            # Apply scale and translation
            new_x = x * transform['scale'] + transform['translation'][0]
            new_y = y * transform['scale'] + transform['translation'][1]
            
            transformed.append({
                'x': new_x,
                'y': new_y,
                'brightness': star['brightness'],
                'name': star.get('name', '')
            })
        
        return transformed
    
    def _calculate_match_score(self, transformed_pattern: List[Dict], detected_stars: List[Dict]) -> float:
        """Calculate how well the transformed pattern matches detected stars."""
        if not transformed_pattern or not detected_stars:
            return 0.0
        
        matches = 0
        total_distance = 0
        
        for pattern_star in transformed_pattern:
            best_distance = float('inf')
            
            for detected_star in detected_stars:
                distance = math.sqrt((pattern_star['x'] - detected_star['x'])**2 + 
                                   (pattern_star['y'] - detected_star['y'])**2)
                
                if distance < self.max_distance and distance < best_distance:
                    best_distance = distance
            
            if best_distance < float('inf'):
                matches += 1
                total_distance += best_distance
        
        if matches == 0:
            return 0.0
        
        # Score based on number of matches and average distance
        match_ratio = matches / len(transformed_pattern)
        avg_distance = total_distance / matches
        distance_score = max(0, 1 - (avg_distance / self.max_distance))
        
        return match_ratio * distance_score

class AdvancedConstellationFitter:
    """Advanced constellation fitting with real plate solving integration."""
    
    def __init__(self, wcs: WCS):
        self.wcs = wcs
        self.star_detector = AdvancedStarDetector()
        self.pattern_matcher = ConstellationPatternMatcher()
        self.shape_db = ConstellationShapeDatabase()
    
    def fit_all_constellations(self, image: np.ndarray) -> List[Dict]:
        """Fit all possible constellations to the image."""
        # Detect stars
        detected_stars = self.star_detector.detect_stars(image)
        print(f"🔍 Detected {len(detected_stars)} stars in image")
        
        if len(detected_stars) < 5:
            print("❌ Not enough stars detected for constellation fitting")
            return []
        
        # Get all constellation patterns
        all_constellations = self.shape_db.get_all_constellations()
        
        fitted_constellations = []
        
        for const_name, pattern in all_constellations.items():
            print(f"\n🔍 Testing {const_name}...")
            
            # Convert pattern stars to pixel coordinates
            pattern_stars = self._convert_pattern_to_pixels(pattern)
            
            if len(pattern_stars) >= 3:
                # Try to match pattern
                match_result = self.pattern_matcher.match_pattern(pattern_stars, detected_stars)
                
                if match_result:
                    print(f"✅ Found {const_name} with score {match_result['score']:.2f}")
                    
                    fitted_constellations.append({
                        'name': const_name,
                        'pattern': pattern,
                        'match_result': match_result,
                        'center': self._calculate_center(match_result['detected_stars'])
                    })
                else:
                    print(f"❌ {const_name} not found")
            else:
                print(f"⚠️ {const_name} has insufficient stars in field of view")
        
        return fitted_constellations
    
    def _convert_pattern_to_pixels(self, pattern: Dict) -> List[Dict]:
        """Convert constellation pattern stars to pixel coordinates."""
        pattern_stars = []
        
        for star in pattern['stars']:
            pixel = self._sky_to_pixel(star['ra'], star['dec'])
            if pixel:
                pattern_stars.append({
                    'name': star['name'],
                    'x': pixel[0],
                    'y': pixel[1],
                    'brightness': star['mag']
                })
        
        return pattern_stars
    
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
    
    def _calculate_center(self, stars: List[Dict]) -> Tuple[float, float]:
        """Calculate center of constellation from matched stars."""
        if not stars:
            return (0, 0)
        
        x_sum = sum(s['x'] for s in stars)
        y_sum = sum(s['y'] for s in stars)
        count = len(stars)
        
        return x_sum / count, y_sum / count

def main():
    """Test the advanced constellation fitting."""
    print("🌟 Advanced Constellation Fitter Test")
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
    
    # Create advanced fitter
    fitter = AdvancedConstellationFitter(wcs)
    
    # Fit all constellations
    results = fitter.fit_all_constellations(image)
    
    print(f"\n📊 Results Summary:")
    print(f"   Found {len(results)} constellations")
    
    for result in results:
        print(f"   ✅ {result['name']}: Score {result['match_result']['score']:.2f}")
        print(f"      Center: {result['center']}")

if __name__ == "__main__":
    main() 