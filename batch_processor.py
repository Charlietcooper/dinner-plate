#!/usr/bin/env python3
"""
Batch Processor for Dinner Plate Constellation Annotator

This script processes all images in the Input folder and saves results to Output folder.
Intermediate files are saved to the Processing folder.

Usage:
    python batch_processor.py
    python batch_processor.py --verbose
    python batch_processor.py --resize-only
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np

from constellation_annotator import ConstellationAnnotator
from resize_image import resize_image
from test_image_analysis import analyze_image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BatchProcessor:
    """Batch processor for constellation annotation."""
    
    def __init__(self, input_dir: str = "Input", processing_dir: str = "Processing", 
                 output_dir: str = "Output", api_key: Optional[str] = None, 
                 camera_config: str = "canon_200d"):
        self.input_dir = Path(input_dir)
        self.processing_dir = Path(processing_dir)
        self.output_dir = Path(output_dir)
        self.api_key = api_key
        self.camera_config = camera_config
        
        # Create directories if they don't exist
        self.input_dir.mkdir(exist_ok=True)
        self.processing_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize annotator with camera config
        self.annotator = ConstellationAnnotator(api_key, camera_config)
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.fits', '.fits.gz'}
    
    def get_input_images(self) -> List[Path]:
        """Get all supported image files from input directory."""
        images = []
        for file_path in self.input_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                images.append(file_path)
        return sorted(images)
    
    def analyze_input_image(self, image_path: Path) -> bool:
        """Analyze an input image to check if it's suitable for processing."""
        logger.info(f"🔍 Analyzing {image_path.name}...")
        
        # Create analysis output path
        analysis_output = self.processing_dir / f"analysis_{image_path.stem}.jpg"
        
        # Run analysis
        try:
            success = analyze_image(str(image_path))
            if success:
                logger.info(f"✅ Analysis completed for {image_path.name}")
                return True
            else:
                logger.warning(f"⚠️ Analysis failed for {image_path.name}")
                return False
        except Exception as e:
            logger.error(f"❌ Analysis error for {image_path.name}: {e}")
            return False
    
    def resize_if_needed(self, image_path: Path) -> Optional[Path]:
        """Resize image if it's too large for plate solving."""
        logger.info(f"📏 Checking size of {image_path.name}...")
        
        # Get image dimensions
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"❌ Could not load {image_path.name}")
            return None
        
        height, width = image.shape[:2]
        max_dimension = 2048
        
        if width > max_dimension or height > max_dimension:
            logger.info(f"📐 Resizing {image_path.name} ({width}x{height} -> max {max_dimension})")
            
            # Create resized output path
            resized_path = self.processing_dir / f"resized_{image_path.name}"
            
            # Resize image
            success = resize_image(str(image_path), str(resized_path), max_dimension)
            if success:
                logger.info(f"✅ Resized image saved: {resized_path.name}")
                return resized_path
            else:
                logger.error(f"❌ Failed to resize {image_path.name}")
                return None
        else:
            logger.info(f"✅ {image_path.name} is already appropriately sized")
            return image_path
    
    def process_image(self, image_path: Path, verbose: bool = False) -> bool:
        """Process a single image through the constellation annotator."""
        logger.info(f"🌟 Processing {image_path.name}...")
        
        # Create output path
        output_path = self.output_dir / f"annotated_{image_path.name}"
        
        # Process with constellation annotator
        try:
            if verbose:
                success = self.annotator.annotate_image(str(image_path), str(output_path))
            else:
                # Suppress verbose output for batch processing
                import logging
                original_level = logging.getLogger().level
                logging.getLogger().setLevel(logging.WARNING)
                
                success = self.annotator.annotate_image(str(image_path), str(output_path))
                
                logging.getLogger().setLevel(original_level)
            
            if success:
                logger.info(f"✅ Successfully processed {image_path.name}")
                logger.info(f"📸 Output saved: {output_path.name}")
                return True
            else:
                logger.error(f"❌ Failed to process {image_path.name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error processing {image_path.name}: {e}")
            return False
    
    def run_batch(self, verbose: bool = False, resize_only: bool = False) -> None:
        """Run batch processing on all input images."""
        logger.info("🚀 Starting batch processing...")
        logger.info(f"📁 Input directory: {self.input_dir}")
        logger.info(f"🔧 Processing directory: {self.processing_dir}")
        logger.info(f"📸 Output directory: {self.output_dir}")
        
        # Get input images
        input_images = self.get_input_images()
        
        if not input_images:
            logger.warning("⚠️ No supported images found in input directory")
            logger.info(f"Supported formats: {', '.join(self.supported_formats)}")
            return
        
        logger.info(f"📊 Found {len(input_images)} images to process")
        
        # Process each image
        successful = 0
        failed = 0
        
        for i, image_path in enumerate(input_images, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing image {i}/{len(input_images)}: {image_path.name}")
            logger.info(f"{'='*60}")
            
            try:
                # Step 1: Analyze image
                if not self.analyze_input_image(image_path):
                    logger.warning(f"⚠️ Skipping {image_path.name} due to analysis issues")
                    failed += 1
                    continue
                
                # Step 2: Resize if needed
                processed_path = self.resize_if_needed(image_path)
                if not processed_path:
                    logger.error(f"❌ Failed to prepare {image_path.name}")
                    failed += 1
                    continue
                
                # Step 3: Process with constellation annotator (unless resize-only mode)
                if not resize_only:
                    if self.process_image(processed_path, verbose):
                        successful += 1
                    else:
                        failed += 1
                else:
                    logger.info(f"✅ Resize-only mode: {image_path.name} prepared")
                    successful += 1
                    
            except Exception as e:
                logger.error(f"❌ Unexpected error processing {image_path.name}: {e}")
                failed += 1
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("📊 BATCH PROCESSING SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"✅ Successful: {successful}")
        logger.info(f"❌ Failed: {failed}")
        logger.info(f"📊 Total: {len(input_images)}")
        
        if successful > 0:
            logger.info(f"\n🎉 Processing complete! Check the Output folder for results.")
        else:
            logger.warning(f"\n⚠️ No images were successfully processed.")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Batch processor for Dinner Plate constellation annotator")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose output")
    parser.add_argument("--resize-only", action="store_true",
                       help="Only resize images, don't run constellation annotation")
    parser.add_argument("--input-dir", default="Input",
                       help="Input directory (default: Input)")
    parser.add_argument("--processing-dir", default="Processing",
                       help="Processing directory (default: Processing)")
    parser.add_argument("--output-dir", default="Output",
                       help="Output directory (default: Output)")
    parser.add_argument("--api-key",
                       help="Astrometry.net API key (or set ASTROMETRY_API_KEY env var)")
    parser.add_argument("--camera", default="canon_200d",
                       help="Camera configuration (default: canon_200d)")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.environ.get('ASTROMETRY_API_KEY')
    if not api_key:
        logger.warning("⚠️ No API key provided. Using environment variable or config.")
    
    # Create processor
    processor = BatchProcessor(
        input_dir=args.input_dir,
        processing_dir=args.processing_dir,
        output_dir=args.output_dir,
        api_key=api_key,
        camera_config=args.camera
    )
    
    # Run batch processing
    processor.run_batch(verbose=args.verbose, resize_only=args.resize_only)

if __name__ == "__main__":
    main() 