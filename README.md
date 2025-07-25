# Dinner Plate - Constellation Annotator for Wide-Field Astronomical Images

A Python script that automatically draws constellation lines on wide-field astronomical images taken with DSLR cameras. The script uses plate solving to determine the celestial coordinates of the image and then overlays constellation patterns.

**Project Name**: Dinner Plate (because it's like putting constellations on a dinner plate-sized view of the sky!)

## Features

- **Plate Solving**: Uses Astrometry.net to determine the World Coordinate System (WCS) of your image
- **Wide-Field Support**: Designed specifically for large portions of the sky
- **Multiple Formats**: Supports JPEG, PNG, and FITS image formats
- **Automatic Detection**: Identifies which constellations are visible in your image
- **Clean Output**: Draws constellation lines and labels on the original image
- **File-in-File-out**: Simple command-line interface for batch processing

## Installation

### Prerequisites

- Python 3.7 or higher
- An Astrometry.net API key (free at https://nova.astrometry.net/)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/Charlietcooper/dinner-plate.git
cd dinner-plate

# Run the setup script
python setup.py

# Or install dependencies manually
pip install -r requirements.txt
```

### Get an API Key

1. Visit https://nova.astrometry.net/
2. Create a free account
3. Go to your profile and copy your API key
4. Set it as an environment variable:
   ```bash
   export ASTROMETRY_API_KEY="your_api_key_here"
   ```

## Usage

### Basic Usage

```bash
python constellation_annotator.py input_image.jpg output_image.jpg
```

### With API Key

```bash
python constellation_annotator.py input_image.jpg output_image.jpg --api-key YOUR_API_KEY
```

### Verbose Output

```bash
python constellation_annotator.py input_image.jpg output_image.jpg --verbose
```

### Examples

```bash
# Process a JPEG image
python constellation_annotator.py milky_way.jpg milky_way_annotated.jpg

# Process a FITS file
python constellation_annotator.py observation.fits observation_annotated.jpg

# Process with verbose logging
python constellation_annotator.py night_sky.jpg night_sky_annotated.jpg --verbose
```

## How It Works

1. **Image Loading**: The script loads your astronomical image (JPEG, PNG, or FITS)
2. **Plate Solving**: Uses Astrometry.net to determine the celestial coordinates of each pixel
3. **Star Matching**: Identifies bright stars in your image using known star catalogs
4. **Constellation Detection**: Determines which constellations are visible based on star positions
5. **Line Drawing**: Draws connecting lines between stars to form constellation patterns
6. **Labeling**: Adds constellation names to the image
7. **Output**: Saves the annotated image

## Supported Constellations

The script currently includes the following constellations with their brightest stars:

- **Orion**: Betelgeuse, Bellatrix, Mintaka, Alnilam, Alnitak, Saiph, Rigel
- **Ursa Major**: Dubhe, Merak, Phecda, Megrez, Alioth, Mizar, Alkaid
- **Cassiopeia**: Schedar, Caph, Cih, Ruchbah, Segin
- **Cygnus**: Deneb, Sadr, Gienah, Delta Cygni, Albireo
- **Scorpius**: Antares, Shaula, Lesath, Dschubba, Acrab
- **Sagittarius**: Kaus Australis, Kaus Media, Kaus Borealis, Nunki, Ascella
- **And many more...** (30+ constellations total)

## Technical Details

### Plate Solving

The script uses Astrometry.net's blind plate solving service, which:
- Analyzes star patterns in your image
- Matches them against a comprehensive star catalog
- Determines the exact orientation and scale of your image
- Provides a World Coordinate System (WCS) solution

### Coordinate Transformation

Once the WCS is determined, the script:
- Converts known star coordinates (RA/Dec) to pixel positions
- Draws lines between stars that form constellation patterns
- Ensures all annotations are within the image bounds

### Image Processing

- **FITS Support**: Automatically normalizes FITS data to visible range
- **Color Handling**: Preserves original image colors while adding annotations
- **Text Rendering**: Uses PIL for high-quality text labels

## Troubleshooting

### Plate Solving Fails

- **Check API Key**: Ensure your Astrometry.net API key is valid
- **Image Quality**: Make sure your image has enough stars visible
- **File Size**: Very large images may take longer or fail
- **Network**: Ensure you have internet access for the API call

### No Constellations Found

- **Field of View**: The image may not contain any of the supported constellations
- **Star Visibility**: Very faint or overexposed images may not work well
- **Coordinate Range**: Check if your image covers areas with known bright stars

### Poor Quality Output

- **Image Resolution**: Higher resolution images generally work better
- **Star Detection**: Ensure stars are clearly visible and not overexposed
- **Noise**: Very noisy images may affect plate solving accuracy

## Customization

### Adding More Constellations

You can extend the constellation database by modifying the `ConstellationData` class in the script:

```python
def _load_constellation_lines(self):
    # Add your constellation definitions here
    return {
        'Your Constellation': [
            ('Star1', 'Star2'),
            ('Star2', 'Star3'),
            # ... more star pairs
        ],
        # ... existing constellations
    }
```

### Changing Visual Style

Modify the drawing parameters in the `ConstellationAnnotator` class:

```python
# Line appearance
self.line_color = (0, 255, 0)      # Green lines
self.line_thickness = 2            # Line width

# Star markers
self.star_color = (255, 255, 0)    # Yellow stars
self.star_radius = 3               # Star size

# Text labels
self.text_color = (255, 255, 255)  # White text
self.font_size = 16                # Font size
```

## Performance Tips

- **Image Size**: Larger images take longer to process
- **API Limits**: Astrometry.net has rate limits for free accounts
- **Batch Processing**: Process multiple images sequentially to avoid API limits
- **Caching**: The script doesn't cache plate solutions, so repeated runs will re-solve

## Limitations

- **API Dependency**: Requires internet connection and Astrometry.net service
- **Star Catalog**: Limited to bright stars in major constellations
- **Image Quality**: Works best with clear, well-exposed images
- **Field of View**: Very wide fields may have distortion at edges

## Future Enhancements

- Local plate solving without API dependency
- More comprehensive star catalogs
- Additional constellation patterns
- Custom line styles and colors
- Batch processing capabilities
- Integration with other astronomy software

## Contributing

Contributions are welcome! Areas for improvement:
- Additional constellation data
- Better star detection algorithms
- Local plate solving capabilities
- Performance optimizations
- Documentation improvements

## License

This project is open source. Feel free to modify and distribute as needed.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Ensure your API key is valid
3. Try with a different image to isolate the problem
4. Check the verbose output for detailed error messages

## Repository

- **GitHub**: https://github.com/Charlietcooper/dinner-plate
- **Project Name**: Dinner Plate
- **Description**: Constellation annotation for wide-field astronomical images
