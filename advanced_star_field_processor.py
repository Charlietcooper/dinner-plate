#!/usr/bin/env python3
"""
Advanced Star Field Processor - Extracts stars, removes nebulosity, determines FOV
Prepares images for machine learning constellation pattern recognition
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import time
from typing import List, Tuple, Dict, Optional
from scipy import ndimage
from scipy.signal import find_peaks
import json
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedStarFieldProcessor:
    """Advanced star field processing for constellation pattern recognition."""
    
    def __init__(self):
        self.min_star_size = 2
        self.max_star_size = 20
        self.star_threshold = 0.8
        self.nebulosity_removal_strength = 0.3
        self.max_pattern_candidates = 10000  # Limit to prevent slowdown
        
    def extract_star_field(self, image_path: str) -> Tuple[np.ndarray, List[Dict]]:
        """Extract star field by removing nebulosity and detecting stars."""
        logger.info("🌟 Extracting star field from image...")
        
        # Load image
        logger.info(f"   Loading image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        logger.info(f"   Original image size: {image.shape[1]}x{image.shape[0]}")
        
        # Convert to grayscale
        logger.info("   Converting to grayscale...")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Remove nebulosity using high-pass filtering
        logger.info("   Removing nebulosity...")
        nebulosity_removed = self._remove_nebulosity(gray)
        
        # Detect stars
        logger.info("   Detecting stars...")
        stars = self._detect_stars(nebulosity_removed)
        
        # Create star field image
        logger.info("   Creating star field image...")
        star_field = self._create_star_field_image(nebulosity_removed, stars)
        
        logger.info(f"   Detected {len(stars)} stars")
        
        return star_field, stars
    
    def _remove_nebulosity(self, gray_image: np.ndarray) -> np.ndarray:
        """Remove nebulosity using high-pass filtering."""
        # Create a large Gaussian kernel for nebulosity
        kernel_size = int(min(gray_image.shape) * 0.1)  # 10% of image size
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        logger.info(f"   Using kernel size: {kernel_size}x{kernel_size}")
        
        # Apply Gaussian blur to get nebulosity
        nebulosity = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)
        
        # Subtract nebulosity from original
        star_field = cv2.addWeighted(gray_image, 1.0, nebulosity, -self.nebulosity_removal_strength, 0)
        
        # Normalize
        star_field = np.clip(star_field, 0, 255).astype(np.uint8)
        
        return star_field
    
    def _detect_stars(self, star_field: np.ndarray) -> List[Dict]:
        """Detect stars using multiple methods."""
        stars = []
        
        logger.info("   Method 1: Threshold-based detection...")
        # Method 1: Threshold-based detection
        _, thresh = cv2.threshold(star_field, int(255 * self.star_threshold), 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        logger.info(f"   Found {len(contours)} contours")
        
        for contour in tqdm(contours, desc="   Processing contours"):
            area = cv2.contourArea(contour)
            if self.min_star_size < area < self.max_star_size:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Calculate brightness
                    y1, y2 = max(0, cy-2), min(star_field.shape[0], cy+3)
                    x1, x2 = max(0, cx-2), min(star_field.shape[1], cx+3)
                    region = star_field[y1:y2, x1:x2]
                    brightness = np.mean(region) if region.size > 0 else 0
                    
                    stars.append({
                        "x": cx,
                        "y": cy,
                        "brightness": brightness,
                        "area": area,
                        "method": "threshold"
                    })
        
        logger.info(f"   Threshold method found {len(stars)} stars")
        
        logger.info("   Method 2: Peak detection...")
        # Method 2: Peak detection
        peaks = self._find_brightness_peaks(star_field)
        logger.info(f"   Found {len(peaks)} peaks")
        
        for peak in tqdm(peaks, desc="   Processing peaks"):
            x, y = peak
            brightness = star_field[y, x]
            area = self._estimate_star_area(star_field, x, y)
            
            if self.min_star_size < area < self.max_star_size:
                stars.append({
                    "x": x,
                    "y": y,
                    "brightness": brightness,
                    "area": area,
                    "method": "peak"
                })
        
        logger.info(f"   Peak method found {len([s for s in stars if s['method'] == 'peak'])} stars")
        
        # Remove duplicates (stars detected by both methods)
        logger.info("   Removing duplicate detections...")
        unique_stars = self._remove_duplicate_stars(stars)
        logger.info(f"   Final unique stars: {len(unique_stars)}")
        
        return unique_stars
    
    def _find_brightness_peaks(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """Find brightness peaks in the image."""
        # Apply Gaussian smoothing
        smoothed = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Find local maxima
        peaks = []
        threshold = int(255 * self.star_threshold)
        
        # Sample every 5th pixel to speed up processing
        step = 5
        for y in tqdm(range(1, smoothed.shape[0] - 1, step), desc="   Scanning for peaks"):
            for x in range(1, smoothed.shape[1] - 1, step):
                pixel = smoothed[y, x]
                neighborhood = smoothed[y-1:y+2, x-1:x+2]
                
                if pixel == np.max(neighborhood) and pixel > threshold:
                    peaks.append((x, y))
        
        return peaks
    
    def _estimate_star_area(self, image: np.ndarray, x: int, y: int) -> float:
        """Estimate the area of a star at given coordinates."""
        # Count pixels above threshold in a small region
        region_size = 10
        y1 = max(0, y - region_size)
        y2 = min(image.shape[0], y + region_size)
        x1 = max(0, x - region_size)
        x2 = min(image.shape[1], x + region_size)
        
        region = image[y1:y2, x1:x2]
        threshold = int(255 * self.star_threshold)
        area = np.sum(region > threshold)
        
        return area
    
    def _remove_duplicate_stars(self, stars: List[Dict]) -> List[Dict]:
        """Remove duplicate star detections."""
        unique_stars = []
        seen_positions = set()
        
        for star in stars:
            pos = (star["x"], star["y"])
            if pos not in seen_positions:
                seen_positions.add(pos)
                unique_stars.append(star)
        
        return unique_stars
    
    def _create_star_field_image(self, star_field: np.ndarray, stars: List[Dict]) -> np.ndarray:
        """Create a clean star field image."""
        # Create black background
        result = np.zeros_like(star_field)
        
        # Draw detected stars
        for star in tqdm(stars, desc="   Drawing stars"):
            x, y = star["x"], star["y"]
            brightness = int(star["brightness"])  # Ensure integer
            size = max(1, int(star["area"] / 10))
            
            cv2.circle(result, (x, y), size, brightness, -1)
        
        return result
    
    def estimate_field_of_view(self, stars: List[Dict], image_shape: Tuple[int, int]) -> Dict:
        """Estimate field of view based on star distribution."""
        logger.info("🔍 Estimating field of view...")
        
        if len(stars) < 10:
            logger.warning("   Warning: Too few stars for accurate FOV estimation")
            return {"fov_estimate": "unknown", "confidence": "low"}
        
        # Calculate star density
        image_area = image_shape[0] * image_shape[1]
        star_density = len(stars) / image_area
        
        # Analyze star brightness distribution
        brightnesses = [star["brightness"] for star in stars]
        avg_brightness = np.mean(brightnesses)
        std_brightness = np.std(brightnesses)
        
        # Analyze star spacing
        logger.info("   Calculating star spacings...")
        spacings = self._calculate_star_spacings(stars)
        avg_spacing = np.mean(spacings) if spacings else 0
        
        # Estimate FOV based on star density and brightness
        fov_estimate = self._estimate_fov_from_stars(star_density, avg_brightness, avg_spacing, image_shape)
        
        logger.info(f"   Star density: {star_density:.6f} stars/pixel²")
        logger.info(f"   Average brightness: {avg_brightness:.1f}")
        logger.info(f"   Average spacing: {avg_spacing:.1f} pixels")
        logger.info(f"   Estimated FOV: {fov_estimate}")
        
        return {
            "fov_estimate": fov_estimate,
            "star_density": star_density,
            "avg_brightness": avg_brightness,
            "avg_spacing": avg_spacing,
            "num_stars": len(stars),
            "confidence": "medium" if len(stars) > 50 else "low"
        }
    
    def _calculate_star_spacings(self, stars: List[Dict]) -> List[float]:
        """Calculate distances between nearby stars."""
        spacings = []
        
        # Limit to first 100 stars to speed up processing
        sample_stars = stars[:100] if len(stars) > 100 else stars
        
        for i, star1 in enumerate(tqdm(sample_stars, desc="   Calculating spacings")):
            for j, star2 in enumerate(sample_stars[i+1:], i+1):
                dx = star1["x"] - star2["x"]
                dy = star1["y"] - star2["y"]
                distance = np.sqrt(dx*dx + dy*dy)
                
                if distance < 100:  # Only consider nearby stars
                    spacings.append(distance)
        
        return spacings
    
    def _estimate_fov_from_stars(self, density: float, brightness: float, spacing: float, 
                                image_shape: Tuple[int, int]) -> str:
        """Estimate field of view based on star characteristics."""
        width, height = image_shape[1], image_shape[0]
        
        # Very high density, bright stars = wide field
        if density > 0.0001 and brightness > 200:
            return "wide_field (60-90°)"
        
        # High density, medium brightness = medium field
        elif density > 0.00005 and brightness > 150:
            return "medium_field (20-60°)"
        
        # Low density, bright stars = narrow field
        elif density < 0.00001 and brightness > 180:
            return "narrow_field (5-20°)"
        
        # Very low density = very narrow field
        elif density < 0.000005:
            return "very_narrow_field (1-5°)"
        
        else:
            return "unknown_field"
    
    def create_ml_ready_dataset(self, star_field: np.ndarray, stars: List[Dict], 
                               fov_info: Dict) -> Dict:
        """Create machine learning ready dataset."""
        logger.info("🤖 Creating machine learning ready dataset...")
        
        # Extract features for each star
        logger.info("   Extracting star features...")
        star_features = []
        for star in tqdm(stars, desc="   Processing stars"):
            features = {
                "position": (star["x"], star["y"]),
                "brightness": star["brightness"],
                "area": star["area"],
                "relative_brightness": star["brightness"] / 255.0,
                "normalized_x": star["x"] / star_field.shape[1],
                "normalized_y": star["y"] / star_field.shape[0]
            }
            star_features.append(features)
        
        # Create pattern candidates (limited to prevent slowdown)
        logger.info("   Generating pattern candidates...")
        pattern_candidates = self._generate_pattern_candidates(stars)
        
        # Create dataset
        dataset = {
            "image_info": {
                "shape": star_field.shape,
                "fov_estimate": fov_info["fov_estimate"],
                "star_count": len(stars),
                "star_density": fov_info["star_density"]
            },
            "stars": star_features,
            "pattern_candidates": pattern_candidates,
            "metadata": {
                "processing_method": "advanced_star_field_processor",
                "fov_confidence": fov_info["confidence"]
            }
        }
        
        logger.info(f"   Created dataset with {len(star_features)} stars")
        logger.info(f"   Generated {len(pattern_candidates)} pattern candidates")
        
        return dataset
    
    def _generate_pattern_candidates(self, stars: List[Dict]) -> List[Dict]:
        """Generate potential constellation pattern candidates."""
        candidates = []
        
        # Limit number of stars to prevent combinatorial explosion
        max_stars = min(50, len(stars))  # Use at most 50 stars
        sample_stars = stars[:max_stars]
        
        logger.info(f"   Using {max_stars} stars for pattern generation")
        
        # Generate triangles (3-star patterns) - limited
        triangle_count = 0
        max_triangles = 5000
        
        logger.info("   Generating triangle patterns...")
        for i in tqdm(range(len(sample_stars)), desc="   Triangle patterns"):
            if triangle_count >= max_triangles:
                break
                
            for j in range(i+1, len(sample_stars)):
                if triangle_count >= max_triangles:
                    break
                    
                for k in range(j+1, len(sample_stars)):
                    if triangle_count >= max_triangles:
                        break
                        
                    star1, star2, star3 = sample_stars[i], sample_stars[j], sample_stars[k]
                    
                    # Calculate triangle properties
                    sides = self._calculate_triangle_sides(star1, star2, star3)
                    area = self._calculate_triangle_area(sides)
                    
                    if area > 100:  # Minimum area threshold
                        candidate = {
                            "type": "triangle",
                            "stars": [star1, star2, star3],
                            "sides": sides,
                            "area": area,
                            "center": self._calculate_pattern_center([star1, star2, star3])
                        }
                        candidates.append(candidate)
                        triangle_count += 1
        
        logger.info(f"   Generated {triangle_count} triangle patterns")
        
        # Generate lines (2-star patterns) - limited
        line_count = 0
        max_lines = 5000
        
        logger.info("   Generating line patterns...")
        for i in tqdm(range(len(sample_stars)), desc="   Line patterns"):
            if line_count >= max_lines:
                break
                
            for j in range(i+1, len(sample_stars)):
                if line_count >= max_lines:
                    break
                    
                star1, star2 = sample_stars[i], sample_stars[j]
                distance = np.sqrt((star1["x"] - star2["x"])**2 + (star1["y"] - star2["y"])**2)
                
                if 20 < distance < 200:  # Reasonable line length
                    candidate = {
                        "type": "line",
                        "stars": [star1, star2],
                        "length": distance,
                        "center": self._calculate_pattern_center([star1, star2])
                    }
                    candidates.append(candidate)
                    line_count += 1
        
        logger.info(f"   Generated {line_count} line patterns")
        logger.info(f"   Total patterns: {len(candidates)}")
        
        return candidates
    
    def _calculate_triangle_sides(self, star1: Dict, star2: Dict, star3: Dict) -> List[float]:
        """Calculate the sides of a triangle formed by three stars."""
        def distance(s1, s2):
            return np.sqrt((s1["x"] - s2["x"])**2 + (s1["y"] - s2["y"])**2)
        
        return [
            distance(star1, star2),
            distance(star2, star3),
            distance(star3, star1)
        ]
    
    def _calculate_triangle_area(self, sides: List[float]) -> float:
        """Calculate area of triangle using Heron's formula."""
        a, b, c = sides
        s = (a + b + c) / 2
        return np.sqrt(s * (s - a) * (s - b) * (s - c))
    
    def _calculate_pattern_center(self, stars: List[Dict]) -> Tuple[float, float]:
        """Calculate the center of a pattern."""
        x_sum = sum(star["x"] for star in stars)
        y_sum = sum(star["y"] for star in stars)
        return x_sum / len(stars), y_sum / len(stars)
    
    def save_processed_data(self, star_field: np.ndarray, stars: List[Dict], 
                           fov_info: Dict, dataset: Dict, output_dir: str = "Processing"):
        """Save all processed data."""
        logger.info("💾 Saving processed data...")
        
        # Save star field image
        star_field_path = os.path.join(output_dir, "star_field_extracted.jpg")
        cv2.imwrite(star_field_path, star_field)
        logger.info(f"   Star field image: {star_field_path}")
        
        # Save star data (convert numpy types to native Python types)
        stars_serializable = []
        for star in stars:
            star_copy = star.copy()
            star_copy["brightness"] = float(star_copy["brightness"])
            star_copy["area"] = float(star_copy["area"])
            stars_serializable.append(star_copy)
        
        stars_path = os.path.join(output_dir, "detected_stars.json")
        with open(stars_path, 'w') as f:
            json.dump(stars_serializable, f, indent=2)
        logger.info(f"   Star data: {stars_path}")
        
        # Save FOV info
        fov_path = os.path.join(output_dir, "fov_estimation.json")
        with open(fov_path, 'w') as f:
            json.dump(fov_info, f, indent=2)
        logger.info(f"   FOV info: {fov_path}")
        
        # Save ML dataset (convert numpy types)
        dataset_serializable = self._make_json_serializable(dataset)
        dataset_path = os.path.join(output_dir, "ml_dataset.json")
        with open(dataset_path, 'w') as f:
            json.dump(dataset_serializable, f, indent=2)
        logger.info(f"   ML dataset: {dataset_path}")
        
        # Create visualization
        viz_path = os.path.join("Output", "star_field_analysis.jpg")
        self._create_analysis_visualization(star_field, stars, fov_info, viz_path)
        logger.info(f"   Analysis visualization: {viz_path}")
    
    def _create_analysis_visualization(self, star_field: np.ndarray, stars: List[Dict], 
                                     fov_info: Dict, output_path: str):
        """Create a visualization of the star field analysis."""
        # Create a color version of the star field
        viz_image = cv2.cvtColor(star_field, cv2.COLOR_GRAY2BGR)
        
        # Draw detected stars with different colors based on brightness
        for star in tqdm(stars, desc="   Drawing visualization"):
            x, y = star["x"], star["y"]
            brightness = star["brightness"]
            
            # Color based on brightness (BGR format for OpenCV)
            if brightness > 200:
                color = (0, 255, 255)  # Yellow for bright stars
                size = 4
            elif brightness > 150:
                color = (255, 255, 0)  # Cyan for medium stars
                size = 3
            else:
                color = (128, 128, 128)  # Gray for dim stars
                size = 2
            
            cv2.circle(viz_image, (x, y), size, color, -1)
        
        # Add text information
        pil_image = Image.fromarray(cv2.cvtColor(viz_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
            small_font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Add title
        title = "Advanced Star Field Analysis"
        draw.text((20, 20), title, fill=(255, 255, 255), font=font)
        
        # Add analysis info
        info_lines = [
            f"Stars detected: {len(stars)}",
            f"FOV estimate: {fov_info.get('fov_estimate', 'unknown')}",
            f"Star density: {fov_info.get('star_density', 0):.6f} stars/pixel²",
            f"Average brightness: {fov_info.get('avg_brightness', 0):.1f}",
            f"Confidence: {fov_info.get('confidence', 'unknown')}"
        ]
        
        y_offset = 60
        for line in info_lines:
            draw.text((20, y_offset), line, fill=(255, 255, 255), font=small_font)
            y_offset += 25
        
        # Add legend
        legend_y = y_offset + 20
        draw.text((20, legend_y), "Star Brightness:", fill=(255, 255, 255), font=small_font)
        legend_y += 25
        draw.text((20, legend_y), "● Bright (>200)", fill=(0, 255, 255), font=small_font)
        legend_y += 20
        draw.text((20, legend_y), "● Medium (150-200)", fill=(255, 255, 0), font=small_font)
        legend_y += 20
        draw.text((20, legend_y), "● Dim (<150)", fill=(128, 128, 128), font=small_font)
        
        # Convert back and save
        result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, result_image)
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tuple):
            return list(obj)
        else:
            return obj

def main():
    """Test advanced star field processing on the test image."""
    print("🌟 Advanced Star Field Processor Test")
    print("=" * 50)
    print("Testing star extraction, nebulosity removal, and FOV estimation")
    print("=" * 50)
    
    start_time = time.time()
    
    # Create processor
    processor = AdvancedStarFieldProcessor()
    
    # Test image path
    test_image_path = "Input/test-1.jpg"
    
    try:
        # Process image
        star_field, stars = processor.extract_star_field(test_image_path)
        
        # Estimate FOV
        fov_info = processor.estimate_field_of_view(stars, star_field.shape)
        
        # Create ML dataset
        dataset = processor.create_ml_ready_dataset(star_field, stars, fov_info)
        
        # Save results
        processor.save_processed_data(star_field, stars, fov_info, dataset)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ Advanced star field processing complete!")
        print(f"   Extracted {len(stars)} stars")
        print(f"   Estimated FOV: {fov_info['fov_estimate']}")
        print(f"   Created ML-ready dataset")
        print(f"   Ready for machine learning pattern recognition")
        print(f"   Total processing time: {elapsed_time:.1f} seconds")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        raise

if __name__ == "__main__":
    main() 