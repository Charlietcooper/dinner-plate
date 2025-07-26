#!/usr/bin/env python3
"""
Improved Star Field Processor - Better star detection and pattern recognition
Addresses issues with star density, pattern accuracy, and constellation overlay
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

class ImprovedStarFieldProcessor:
    """Improved star field processing with better detection and pattern recognition."""
    
    def __init__(self):
        # Improved parameters for better star detection
        self.min_star_size = 1  # Reduced for more stars
        self.max_star_size = 50  # Increased for larger stars
        self.star_threshold = 0.6  # Lowered for more sensitive detection
        self.nebulosity_removal_strength = 0.4  # Increased removal
        self.brightness_threshold = 100  # Lower brightness threshold
        
    def extract_star_field(self, image_path: str) -> Tuple[np.ndarray, List[Dict]]:
        """Extract star field with improved detection."""
        logger.info("🌟 Improved star field extraction...")
        
        # Load image
        logger.info(f"   Loading image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        logger.info(f"   Original image size: {image.shape[1]}x{image.shape[0]}")
        
        # Convert to grayscale
        logger.info("   Converting to grayscale...")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Enhanced nebulosity removal
        logger.info("   Enhanced nebulosity removal...")
        nebulosity_removed = self._enhanced_nebulosity_removal(gray)
        
        # Multi-method star detection
        logger.info("   Multi-method star detection...")
        stars = self._enhanced_star_detection(nebulosity_removed)
        
        # Create enhanced star field image
        star_field = self._create_enhanced_star_field(nebulosity_removed, stars)
        
        logger.info(f"   Detected {len(stars)} stars with improved methods")
        
        return star_field, stars
    
    def _enhanced_nebulosity_removal(self, gray_image: np.ndarray) -> np.ndarray:
        """Enhanced nebulosity removal with multiple techniques."""
        # Method 1: High-pass filtering
        kernel_size = int(min(gray_image.shape) * 0.08)  # Smaller kernel
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        logger.info(f"   Using kernel size: {kernel_size}x{kernel_size}")
        
        # Apply Gaussian blur to get nebulosity
        nebulosity = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)
        
        # Subtract nebulosity with increased strength
        star_field = cv2.addWeighted(gray_image, 1.0, nebulosity, -self.nebulosity_removal_strength, 0)
        
        # Method 2: Additional contrast enhancement
        star_field = cv2.equalizeHist(star_field)
        
        # Method 3: Noise reduction
        star_field = cv2.medianBlur(star_field, 3)
        
        # Normalize
        star_field = np.clip(star_field, 0, 255).astype(np.uint8)
        
        return star_field
    
    def _enhanced_star_detection(self, star_field: np.ndarray) -> List[Dict]:
        """Enhanced star detection using multiple methods."""
        stars = []
        
        # Method 1: Adaptive thresholding
        logger.info("   Method 1: Adaptive thresholding...")
        adaptive_thresh = cv2.adaptiveThreshold(
            star_field, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        logger.info(f"   Found {len(contours)} contours with adaptive threshold")
        
        for contour in tqdm(contours, desc="   Processing adaptive contours"):
            area = cv2.contourArea(contour)
            if self.min_star_size < area < self.max_star_size:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Calculate brightness
                    y1, y2 = max(0, cy-3), min(star_field.shape[0], cy+4)
                    x1, x2 = max(0, cx-3), min(star_field.shape[1], cx+4)
                    region = star_field[y1:y2, x1:x2]
                    brightness = np.mean(region) if region.size > 0 else 0
                    
                    if brightness > self.brightness_threshold:
                        stars.append({
                            "x": cx,
                            "y": cy,
                            "brightness": brightness,
                            "area": area,
                            "method": "adaptive_threshold"
                        })
        
        logger.info(f"   Adaptive threshold found {len([s for s in stars if s['method'] == 'adaptive_threshold'])} stars")
        
        # Method 2: Simple thresholding with lower threshold
        logger.info("   Method 2: Simple thresholding...")
        _, thresh = cv2.threshold(star_field, int(255 * self.star_threshold), 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        logger.info(f"   Found {len(contours)} contours with simple threshold")
        
        for contour in tqdm(contours, desc="   Processing simple contours"):
            area = cv2.contourArea(contour)
            if self.min_star_size < area < self.max_star_size:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Calculate brightness
                    y1, y2 = max(0, cy-3), min(star_field.shape[0], cy+4)
                    x1, x2 = max(0, cx-3), min(star_field.shape[1], cx+4)
                    region = star_field[y1:y2, x1:x2]
                    brightness = np.mean(region) if region.size > 0 else 0
                    
                    if brightness > self.brightness_threshold:
                        stars.append({
                            "x": cx,
                            "y": cy,
                            "brightness": brightness,
                            "area": area,
                            "method": "simple_threshold"
                        })
        
        logger.info(f"   Simple threshold found {len([s for s in stars if s['method'] == 'simple_threshold'])} stars")
        
        # Method 3: Peak detection with improved parameters
        logger.info("   Method 3: Enhanced peak detection...")
        peaks = self._enhanced_peak_detection(star_field)
        logger.info(f"   Found {len(peaks)} peaks")
        
        for peak in tqdm(peaks, desc="   Processing peaks"):
            x, y = peak
            brightness = star_field[y, x]
            area = self._estimate_star_area(star_field, x, y)
            
            if self.min_star_size < area < self.max_star_size and brightness > self.brightness_threshold:
                stars.append({
                    "x": x,
                    "y": y,
                    "brightness": brightness,
                    "area": area,
                    "method": "peak_detection"
                })
        
        logger.info(f"   Peak detection found {len([s for s in stars if s['method'] == 'peak_detection'])} stars")
        
        # Remove duplicates and filter by quality
        unique_stars = self._remove_duplicate_stars(stars)
        filtered_stars = self._filter_stars_by_quality(unique_stars)
        
        logger.info(f"   Final unique stars: {len(filtered_stars)}")
        
        return filtered_stars
    
    def _enhanced_peak_detection(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """Enhanced peak detection with better parameters."""
        # Apply Gaussian smoothing
        smoothed = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Find local maxima with lower threshold
        peaks = []
        threshold = int(255 * 0.5)  # Lower threshold for more peaks
        
        # Sample every 3rd pixel for speed
        step = 3
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
        region_size = 15  # Increased region size
        y1 = max(0, y - region_size)
        y2 = min(image.shape[0], y + region_size)
        x1 = max(0, x - region_size)
        x2 = min(image.shape[1], x + region_size)
        
        region = image[y1:y2, x1:x2]
        threshold = int(255 * 0.4)  # Lower threshold
        area = np.sum(region > threshold)
        
        return area
    
    def _remove_duplicate_stars(self, stars: List[Dict]) -> List[Dict]:
        """Remove duplicate star detections with improved logic."""
        unique_stars = []
        seen_positions = set()
        
        for star in stars:
            pos = (star["x"], star["y"])
            if pos not in seen_positions:
                seen_positions.add(pos)
                unique_stars.append(star)
        
        return unique_stars
    
    def _filter_stars_by_quality(self, stars: List[Dict]) -> List[Dict]:
        """Filter stars by quality metrics."""
        if len(stars) < 50:
            return stars  # Keep all if we have few stars
        
        # Sort by brightness and area
        sorted_stars = sorted(stars, key=lambda s: (s["brightness"], s["area"]), reverse=True)
        
        # Keep top 80% of stars
        keep_count = int(len(sorted_stars) * 0.8)
        return sorted_stars[:keep_count]
    
    def _create_enhanced_star_field(self, star_field: np.ndarray, stars: List[Dict]) -> np.ndarray:
        """Create an enhanced star field image."""
        # Create black background
        result = np.zeros_like(star_field)
        
        # Draw detected stars with enhanced visualization
        for star in tqdm(stars, desc="   Drawing enhanced stars"):
            x, y = star["x"], star["y"]
            brightness = int(star["brightness"])
            size = max(1, int(star["area"] / 8))  # Adjusted size calculation
            
            cv2.circle(result, (x, y), size, brightness, -1)
        
        return result
    
    def estimate_field_of_view(self, stars: List[Dict], image_shape: Tuple[int, int]) -> Dict:
        """Estimate field of view with improved analysis."""
        logger.info("🔍 Estimating field of view with improved analysis...")
        
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
        
        # Analyze star spacing with improved calculation
        spacings = self._calculate_improved_star_spacings(stars)
        avg_spacing = np.mean(spacings) if spacings else 0
        
        # Improved FOV estimation
        fov_estimate = self._improved_fov_estimation(star_density, avg_brightness, avg_spacing, image_shape)
        
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
            "confidence": "high" if len(stars) > 100 else "medium"
        }
    
    def _calculate_improved_star_spacings(self, stars: List[Dict]) -> List[float]:
        """Calculate star spacings with improved algorithm."""
        spacings = []
        
        # Use more stars for better statistics
        sample_stars = stars[:200] if len(stars) > 200 else stars
        
        for i, star1 in enumerate(tqdm(sample_stars, desc="   Calculating improved spacings")):
            for j, star2 in enumerate(sample_stars[i+1:], i+1):
                dx = star1["x"] - star2["x"]
                dy = star1["y"] - star2["y"]
                distance = np.sqrt(dx*dx + dy*dy)
                
                if distance < 150:  # Increased range for better statistics
                    spacings.append(distance)
        
        return spacings
    
    def _improved_fov_estimation(self, density: float, brightness: float, spacing: float, 
                                image_shape: Tuple[int, int]) -> str:
        """Improved FOV estimation with better thresholds."""
        width, height = image_shape[1], image_shape[0]
        
        # Adjusted thresholds based on analysis
        if density > 0.00005 and brightness > 180:
            return "wide_field (60-90°)"
        elif density > 0.00002 and brightness > 150:
            return "medium_field (20-60°)"
        elif density > 0.000005 and brightness > 120:
            return "narrow_field (5-20°)"
        elif density < 0.000005:
            return "very_narrow_field (1-5°)"
        else:
            return "medium_field (20-60°)"  # Default to medium field
    
    def create_improved_visualization(self, star_field: np.ndarray, stars: List[Dict], 
                                    fov_info: Dict, output_path: str):
        """Create an improved visualization of the star field analysis."""
        # Create a color version of the star field
        viz_image = cv2.cvtColor(star_field, cv2.COLOR_GRAY2BGR)
        
        # Draw detected stars with improved color coding
        for star in tqdm(stars, desc="   Drawing improved visualization"):
            x, y = star["x"], star["y"]
            brightness = star["brightness"]
            
            # Enhanced color coding based on brightness
            if brightness > 220:
                color = (0, 255, 255)  # Yellow for very bright stars
                size = 5
            elif brightness > 180:
                color = (255, 255, 0)  # Cyan for bright stars
                size = 4
            elif brightness > 140:
                color = (0, 255, 0)  # Green for medium stars
                size = 3
            else:
                color = (128, 128, 128)  # Gray for dim stars
                size = 2
            
            cv2.circle(viz_image, (x, y), size, color, -1)
        
        # Add comprehensive text information
        pil_image = Image.fromarray(cv2.cvtColor(viz_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 24)
            small_font = ImageFont.truetype("arial.ttf", 18)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Add title
        title = "Improved Star Field Analysis"
        draw.text((20, 20), title, fill=(255, 255, 255), font=font)
        
        # Add detailed analysis info
        info_lines = [
            f"Stars detected: {len(stars)}",
            f"FOV estimate: {fov_info.get('fov_estimate', 'unknown')}",
            f"Star density: {fov_info.get('star_density', 0):.6f} stars/pixel²",
            f"Average brightness: {fov_info.get('avg_brightness', 0):.1f}",
            f"Average spacing: {fov_info.get('avg_spacing', 0):.1f} pixels",
            f"Confidence: {fov_info.get('confidence', 'unknown')}"
        ]
        
        y_offset = 70
        for line in info_lines:
            draw.text((20, y_offset), line, fill=(255, 255, 255), font=small_font)
            y_offset += 25
        
        # Add enhanced legend
        legend_y = y_offset + 20
        draw.text((20, legend_y), "Star Brightness (Improved):", fill=(255, 255, 255), font=small_font)
        legend_y += 25
        draw.text((20, legend_y), "● Very Bright (>220)", fill=(0, 255, 255), font=small_font)
        legend_y += 20
        draw.text((20, legend_y), "● Bright (180-220)", fill=(255, 255, 0), font=small_font)
        legend_y += 20
        draw.text((20, legend_y), "● Medium (140-180)", fill=(0, 255, 0), font=small_font)
        legend_y += 20
        draw.text((20, legend_y), "● Dim (<140)", fill=(128, 128, 128), font=small_font)
        
        # Convert back and save
        result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, result_image)

def main():
    """Test improved star field processing."""
    print("🌟 Improved Star Field Processor Test")
    print("=" * 50)
    print("Testing enhanced star detection and pattern recognition")
    print("=" * 50)
    
    start_time = time.time()
    
    # Create processor
    processor = ImprovedStarFieldProcessor()
    
    # Test image path
    test_image_path = "Input/test-1.jpg"
    
    try:
        # Process image
        star_field, stars = processor.extract_star_field(test_image_path)
        
        # Estimate FOV
        fov_info = processor.estimate_field_of_view(stars, star_field.shape)
        
        # Save results
        cv2.imwrite("Processing/improved_star_field.jpg", star_field)
        
        # Save star data
        stars_serializable = []
        for star in stars:
            star_copy = star.copy()
            star_copy["brightness"] = float(star_copy["brightness"])
            star_copy["area"] = float(star_copy["area"])
            stars_serializable.append(star_copy)
        
        with open("Processing/improved_detected_stars.json", 'w') as f:
            json.dump(stars_serializable, f, indent=2)
        
        # Save FOV info
        with open("Processing/improved_fov_estimation.json", 'w') as f:
            json.dump(fov_info, f, indent=2)
        
        # Create improved visualization
        viz_path = os.path.join("Output", "improved_star_field_analysis.jpg")
        processor.create_improved_visualization(star_field, stars, fov_info, viz_path)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ Improved star field processing complete!")
        print(f"   Extracted {len(stars)} stars (improved detection)")
        print(f"   Estimated FOV: {fov_info['fov_estimate']}")
        print(f"   Star density: {fov_info['star_density']:.6f} stars/pixel²")
        print(f"   Average brightness: {fov_info['avg_brightness']:.1f}")
        print(f"   Total processing time: {elapsed_time:.1f} seconds")
        print(f"   Check Output/improved_star_field_analysis.jpg for results")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        raise

if __name__ == "__main__":
    main() 