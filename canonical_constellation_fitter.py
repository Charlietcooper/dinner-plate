#!/usr/bin/env python3
"""
Canonical Constellation Fitter - Preserves constellation shapes with distortion limits
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
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

class CanonicalConstellationShape:
    """Represents a canonical constellation shape with distortion limits."""
    
    def __init__(self, name: str, stars: List[Dict], lines: List[Tuple[str, str]]):
        self.name = name
        self.stars = stars  # List of {"name": str, "ra": float, "dec": float, "mag": float}
        self.lines = lines  # List of (star1_name, star2_name) tuples
        
        # Calculate canonical shape properties
        self.canonical_coords = np.array([[s["ra"], s["dec"]] for s in stars])
        self.canonical_distances = self._calculate_distances()
        self.canonical_angles = self._calculate_angles()
        
        # Distortion limits (tolerance for shape preservation)
        self.distance_tolerance = 0.3  # 30% tolerance for distances
        self.angle_tolerance = 15.0    # 15 degrees tolerance for angles
    
    def _calculate_distances(self) -> np.ndarray:
        """Calculate distances between all star pairs."""
        n_stars = len(self.stars)
        distances = np.zeros((n_stars, n_stars))
        
        for i in range(n_stars):
            for j in range(i+1, n_stars):
                coord1 = self.canonical_coords[i]
                coord2 = self.canonical_coords[j]
                distance = np.sqrt(np.sum((coord1 - coord2)**2))
                distances[i, j] = distance
                distances[j, i] = distance
        
        return distances
    
    def _calculate_angles(self) -> List[float]:
        """Calculate angles between star triplets."""
        angles = []
        n_stars = len(self.stars)
        
        for i in range(n_stars):
            for j in range(n_stars):
                if i == j:
                    continue
                for k in range(n_stars):
                    if k == i or k == j:
                        continue
                    
                    # Calculate angle at star j
                    v1 = self.canonical_coords[i] - self.canonical_coords[j]
                    v2 = self.canonical_coords[k] - self.canonical_coords[j]
                    
                    # Normalize vectors
                    v1_norm = v1 / np.linalg.norm(v1)
                    v2_norm = v2 / np.linalg.norm(v2)
                    
                    # Calculate angle
                    cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
                    angle = np.arccos(cos_angle) * 180 / np.pi
                    angles.append(angle)
        
        return angles

class CanonicalConstellationDatabase:
    """Database of canonical constellation shapes."""
    
    def __init__(self):
        self.constellations = self._create_canonical_constellations()
    
    def _create_canonical_constellations(self) -> Dict[str, CanonicalConstellationShape]:
        """Create canonical constellation shapes with real coordinates."""
        return {
            "Crux": CanonicalConstellationShape(
                "Crux",
                [
                    {"name": "Acrux", "ra": 186.6495, "dec": -63.0991, "mag": 0.77},
                    {"name": "Mimosa", "ra": 191.9303, "dec": -59.6888, "mag": 1.25},
                    {"name": "Gacrux", "ra": 187.7915, "dec": -57.1138, "mag": 1.59},
                    {"name": "Delta Crucis", "ra": 183.7863, "dec": -58.7489, "mag": 2.79}
                ],
                [
                    ("Acrux", "Mimosa"),
                    ("Mimosa", "Gacrux"),
                    ("Gacrux", "Delta Crucis"),
                    ("Delta Crucis", "Acrux")
                ]
            ),
            "Carina": CanonicalConstellationShape(
                "Carina",
                [
                    {"name": "Canopus", "ra": 95.9879, "dec": -52.6957, "mag": -0.74},
                    {"name": "Miaplacidus", "ra": 138.2999, "dec": -69.7172, "mag": 1.67},
                    {"name": "Avior", "ra": 139.2725, "dec": -59.5092, "mag": 1.86},
                    {"name": "Aspidiske", "ra": 139.2725, "dec": -59.5092, "mag": 2.21}
                ],
                [
                    ("Canopus", "Miaplacidus"),
                    ("Miaplacidus", "Avior"),
                    ("Avior", "Aspidiske")
                ]
            ),
            "Vela": CanonicalConstellationShape(
                "Vela",
                [
                    {"name": "Suhail", "ra": 136.9990, "dec": -43.4326, "mag": 1.83},
                    {"name": "Markeb", "ra": 140.5284, "dec": -55.0107, "mag": 2.47},
                    {"name": "Alsephina", "ra": 127.5669, "dec": -49.4201, "mag": 1.75}
                ],
                [
                    ("Suhail", "Markeb"),
                    ("Markeb", "Alsephina"),
                    ("Alsephina", "Suhail")
                ]
            )
        }
    
    def get_constellation(self, name: str) -> Optional[CanonicalConstellationShape]:
        """Get canonical constellation shape by name."""
        return self.constellations.get(name)
    
    def get_all_constellations(self) -> Dict[str, CanonicalConstellationShape]:
        """Get all canonical constellation shapes."""
        return self.constellations

class ShapePreservingFitter:
    """Fits constellation shapes while preserving their canonical form."""
    
    def __init__(self, wcs: WCS):
        self.wcs = wcs
        self.database = CanonicalConstellationDatabase()
    
    def fit_constellation(self, image: np.ndarray, constellation_name: str) -> Optional[Dict]:
        """Fit a constellation while preserving its canonical shape."""
        canonical = self.database.get_constellation(constellation_name)
        if not canonical:
            return None
        
        # Detect stars in image
        detected_stars = self._detect_stars(image)
        if len(detected_stars) < len(canonical.stars):
            return None
        
        # Try to find the best fit that preserves shape
        best_fit = self._find_shape_preserving_fit(canonical, detected_stars)
        
        if best_fit:
            return {
                "constellation": constellation_name,
                "canonical": canonical,
                "detected_matches": best_fit["matches"],
                "transform": best_fit["transform"],
                "shape_score": best_fit["shape_score"]
            }
        
        return None
    
    def _detect_stars(self, image: np.ndarray) -> List[Dict]:
        """Detect stars in the image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Simple star detection
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        stars = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 5 <= area <= 100:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    brightness = int(gray[cy, cx])
                    
                    stars.append({
                        "x": cx, "y": cy, "brightness": brightness, "area": area
                    })
        
        # Sort by brightness and limit to top stars
        stars.sort(key=lambda x: x['brightness'], reverse=True)
        return stars[:50]
    
    def _find_shape_preserving_fit(self, canonical: CanonicalConstellationShape, 
                                 detected_stars: List[Dict]) -> Optional[Dict]:
        """Find the best fit that preserves the canonical shape."""
        best_fit = None
        best_score = 0
        
        # Try different combinations of detected stars
        from itertools import combinations
        
        for detected_combo in combinations(detected_stars, len(canonical.stars)):
            # Try different orderings of canonical stars
            from itertools import permutations
            
            for canonical_perm in permutations(range(len(canonical.stars))):
                fit_result = self._test_shape_fit(canonical, detected_combo, canonical_perm)
                
                if fit_result and fit_result["shape_score"] > best_score:
                    best_fit = fit_result
                    best_score = fit_result["shape_score"]
        
        return best_fit if best_score > 0.7 else None  # Minimum 70% shape preservation
    
    def _test_shape_fit(self, canonical: CanonicalConstellationShape, 
                       detected_stars: List[Dict], 
                       canonical_perm: Tuple[int, ...]) -> Optional[Dict]:
        """Test if a specific fit preserves the canonical shape."""
        # Get canonical coordinates in the test permutation
        canonical_coords = canonical.canonical_coords[list(canonical_perm)]
        
        # Get detected coordinates
        detected_coords = np.array([[s["x"], s["y"]] for s in detected_stars])
        
        # Calculate transformation parameters
        transform = self._calculate_transformation(canonical_coords, detected_coords)
        if not transform:
            return None
        
        # Apply transformation to canonical coordinates
        transformed_canonical = self._apply_transformation(canonical_coords, transform)
        
        # Calculate shape preservation score
        shape_score = self._calculate_shape_score(canonical, transformed_canonical, detected_coords)
        
        if shape_score > 0.5:  # Minimum shape preservation
            return {
                "matches": list(zip(canonical_perm, range(len(detected_stars)))),
                "transform": transform,
                "shape_score": shape_score,
                "transformed_coords": transformed_canonical
            }
        
        return None
    
    def _calculate_transformation(self, canonical_coords: np.ndarray, 
                                detected_coords: np.ndarray) -> Optional[Dict]:
        """Calculate transformation between canonical and detected coordinates."""
        if len(canonical_coords) < 2:
            return None
        
        # Calculate centroids
        canonical_centroid = np.mean(canonical_coords, axis=0)
        detected_centroid = np.mean(detected_coords, axis=0)
        
        # Center coordinates
        canonical_centered = canonical_coords - canonical_centroid
        detected_centered = detected_coords - detected_centroid
        
        # Calculate scale
        canonical_distances = cdist(canonical_centered, canonical_centered)
        detected_distances = cdist(detected_centered, detected_centered)
        
        # Find scale factor
        scale_factors = []
        for i in range(len(canonical_distances)):
            for j in range(i+1, len(canonical_distances)):
                if canonical_distances[i, j] > 0:
                    scale = detected_distances[i, j] / canonical_distances[i, j]
                    scale_factors.append(scale)
        
        if not scale_factors:
            return None
        
        scale = np.median(scale_factors)
        
        # Check scale tolerance
        if not (0.5 <= scale <= 2.0):  # Allow 50% to 200% scale
            return None
        
        return {
            "scale": scale,
            "translation": detected_centroid - canonical_centroid * scale
        }
    
    def _apply_transformation(self, coords: np.ndarray, transform: Dict) -> np.ndarray:
        """Apply transformation to coordinates."""
        return coords * transform["scale"] + transform["translation"]
    
    def _calculate_shape_score(self, canonical: CanonicalConstellationShape, 
                             transformed_canonical: np.ndarray, 
                             detected_coords: np.ndarray) -> float:
        """Calculate how well the shape is preserved."""
        # Calculate distances in transformed canonical
        transformed_distances = cdist(transformed_canonical, transformed_canonical)
        
        # Calculate distances in detected
        detected_distances = cdist(detected_coords, detected_coords)
        
        # Compare distance ratios
        distance_scores = []
        for i in range(len(transformed_distances)):
            for j in range(i+1, len(transformed_distances)):
                if transformed_distances[i, j] > 0 and detected_distances[i, j] > 0:
                    ratio = transformed_distances[i, j] / detected_distances[i, j]
                    # Score based on how close ratio is to 1.0
                    score = max(0, 1 - abs(ratio - 1) / canonical.distance_tolerance)
                    distance_scores.append(score)
        
        if not distance_scores:
            return 0.0
        
        return np.mean(distance_scores)

def main():
    """Test the canonical constellation fitting."""
    print("🌟 Canonical Constellation Shape Fitter")
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
    
    # Create WCS (would come from real plate solving)
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [image.shape[1]//2, image.shape[0]//2]
    wcs.wcs.crval = [180, -60]  # Southern sky
    wcs.wcs.cdelt = [0.1, 0.1]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    
    # Create fitter
    fitter = ShapePreservingFitter(wcs)
    
    # Test fitting constellations
    test_constellations = ["Crux", "Carina", "Vela"]
    
    for const_name in test_constellations:
        print(f"\n🔍 Testing {const_name} with shape preservation...")
        result = fitter.fit_constellation(image, const_name)
        
        if result:
            print(f"✅ Found {const_name} with shape score {result['shape_score']:.2f}")
            print(f"   Transform: scale={result['transform']['scale']:.2f}")
        else:
            print(f"❌ {const_name} not found or shape not preserved")
    
    print(f"\n🎯 Key Points:")
    print(f"   1. Canonical shapes are stored with real astronomical coordinates")
    print(f"   2. Shape preservation ensures constellations look correct")
    print(f"   3. Distortion limits prevent unrealistic deformations")
    print(f"   4. Real plate solving would provide accurate positioning")

if __name__ == "__main__":
    main() 