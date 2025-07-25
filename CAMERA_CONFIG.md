# Dinner Plate - Camera Configuration Guide

## 📷 **Camera Configuration Feature**

Your Dinner Plate constellation annotator now supports camera-specific configuration to improve plate solving accuracy!

## 🎯 **Why Camera Configuration Matters**

Plate solving accuracy depends on knowing:
- **Sensor size** (affects field of view)
- **Focal length** (affects image scale)
- **Lens characteristics** (affects distortion)

## 📊 **Your Camera: Canon EOS 200D (Rebel SL2)**

**Default Configuration:**
- **Camera**: Canon EOS 200D (Rebel SL2)
- **Lens**: EF-S 18-55mm f/4-5.6 IS STM
- **Sensor**: 22.3mm × 14.9mm (APS-C)
- **Field of View**: 63.6° × 45.0° (at 18mm)
- **Crop Factor**: 1.6x

## 🚀 **How to Use Camera Configuration**

### **Command Line Options:**
```bash
# Use default Canon 200D configuration
python batch_processor.py --camera canon_200d

# Use 55mm focal length setting
python batch_processor.py --camera canon_200d_55mm

# Use generic DSLR configuration
python batch_processor.py --camera generic_dslr

# Use full-frame configuration
python batch_processor.py --camera full_frame
```

### **Available Camera Configurations:**

| Camera Config | Description | Focal Length | Field of View |
|---------------|-------------|--------------|---------------|
| `canon_200d` | Canon EOS 200D at 18mm | 18mm | 63.6° × 45.0° |
| `canon_200d_55mm` | Canon EOS 200D at 55mm | 55mm | 22.1° × 15.4° |
| `generic_dslr` | Generic APS-C DSLR | 50mm | 25.4° × 17.0° |
| `full_frame` | Full-frame DSLR | 50mm | 39.6° × 27.0° |

## 🔧 **Plate Solving Parameters**

The system automatically calculates optimal plate solving parameters:

- **Scale Estimation**: Based on focal length and sensor size
- **Search Radius**: Based on field of view
- **Scale Bounds**: Upper and lower limits for scale search

### **Example for Canon 200D at 18mm:**
```
Scale Estimate: 57.2 arcseconds/pixel
Scale Range: 28.6 - 114.4 arcseconds/pixel
Search Radius: 31.8 degrees
```

## 📝 **Adding Your Own Camera**

To add a new camera configuration, edit `camera_config.py`:

```python
CAMERA_CONFIGS = {
    "your_camera": {
        "camera": "Your Camera Model",
        "lens": "Your Lens Model",
        "sensor_width_mm": 23.5,  # Your sensor width
        "sensor_height_mm": 15.6,  # Your sensor height
        "focal_length_mm": 50,     # Your focal length
        "aperture": 5.6,           # Your aperture
        "crop_factor": 1.5,        # Your crop factor
    },
}
```

## 🎯 **Best Practices**

### **For Wide-Field Astrophotography:**
- Use **wide-angle lenses** (18-35mm) for Milky Way shots
- Set camera config to match your actual focal length
- Ensure proper exposure and focus

### **For Different Focal Lengths:**
- **18mm**: Great for Milky Way panoramas
- **35mm**: Good for constellation groups
- **50mm+**: Better for individual constellations

### **Plate Solving Tips:**
- **Clear star patterns** work best
- **Avoid light pollution** in images
- **Proper focus** is crucial
- **Good exposure** helps star detection

## 🔍 **Testing Your Configuration**

Test your camera configuration:
```bash
python camera_config.py
```

This will show:
- Camera and lens information
- Calculated field of view
- Plate solving parameters

## 🎉 **Ready to Use!**

Your Canon EOS 200D is now configured and ready! The system will:

1. **Use your camera's specifications** for plate solving
2. **Calculate optimal search parameters** based on your lens
3. **Improve plate solving accuracy** with camera-specific settings
4. **Handle different focal lengths** automatically

Just add your Milky Way photos to the `Input/` folder and run:
```bash
python batch_processor.py --camera canon_200d --verbose
```

The constellation annotator will now work with your specific camera setup! 🌌✨ 