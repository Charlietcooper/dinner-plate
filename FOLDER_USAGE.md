# Dinner Plate - Folder Structure Usage Guide

## 📁 **Folder Organization**

Your Dinner Plate constellation annotator uses a clean 3-folder structure:

```
Astro-Plate-Solve/
├── Input/          # 📥 Place your astronomical images here
├── Processing/     # 🔧 Intermediate files and analysis
└── Output/         # 📸 Final annotated images
```

## 🚀 **How to Use**

### **Step 1: Add Your Images**
Place your astronomical images in the `Input/` folder:
```bash
# Copy your Milky Way photos here
cp /path/to/your/milky_way_photo.jpg Input/
cp /path/to/your/star_field.png Input/
```

### **Step 2: Run Batch Processing**
```bash
# Process all images in Input folder
python batch_processor.py

# With verbose output (see what's happening)
python batch_processor.py --verbose

# Only resize images (skip plate solving)
python batch_processor.py --resize-only
```

### **Step 3: Check Results**
- **Input/**: Your original images
- **Processing/**: Analysis previews, resized images, intermediate files
- **Output/**: Final annotated images with constellation lines

## 📊 **What Each Folder Contains**

### **Input/** 📥
- **Purpose**: Place your astronomical images here
- **Supported formats**: `.jpg`, `.jpeg`, `.png`, `.fits`, `.fits.gz`
- **Example**: `milky_way_photo.jpg`, `star_field.png`

### **Processing/** 🔧
- **Purpose**: Intermediate files and analysis
- **Contains**:
  - `analysis_*.jpg` - Star detection analysis
  - `resized_*.jpg` - Automatically resized images
  - `demo_annotated.jpg` - Example output
  - Other intermediate files

### **Output/** 📸
- **Purpose**: Final annotated images
- **Contains**:
  - `annotated_*.jpg` - Images with constellation lines and labels
  - Ready for viewing and sharing

## 🎯 **Batch Processing Options**

```bash
# Basic processing
python batch_processor.py

# Verbose output (see detailed progress)
python batch_processor.py --verbose

# Only resize images (useful for preparation)
python batch_processor.py --resize-only

# Custom folders
python batch_processor.py --input-dir MyImages --output-dir Results

# With specific API key
python batch_processor.py --api-key YOUR_API_KEY
```

## 📋 **Processing Steps**

For each image, the batch processor:

1. **🔍 Analyzes** the image (checks for stars, brightness, etc.)
2. **📏 Resizes** if too large (for better plate solving)
3. **🌍 Plate solves** using Astrometry.net
4. **⭐ Draws** constellation lines and labels
5. **📸 Saves** the annotated result

## 💡 **Tips for Best Results**

### **Image Requirements:**
- **Clear astronomical images** with visible stars
- **Wide-field photos** (Milky Way, star fields)
- **Proper exposure** (not overexposed/underexposed)
- **Sharp focus** (minimal blur)

### **File Management:**
- Keep original images in `Input/`
- Check `Processing/` for analysis results
- Final results appear in `Output/`
- Clean up `Processing/` folder periodically

### **Troubleshooting:**
- If plate solving fails, check the analysis in `Processing/`
- Try `--resize-only` to prepare images first
- Use `--verbose` to see detailed progress

## 🎉 **Ready to Process Your Images!**

Your folder structure is set up and ready! Just add your Milky Way photos to the `Input/` folder and run:

```bash
python batch_processor.py --verbose
```

The system will automatically process all images and create beautifully annotated versions with constellation lines and labels! 🌌✨ 