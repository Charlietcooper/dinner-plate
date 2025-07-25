# API Keys Guide for Dinner Plate Constellation Annotator

This guide lists all the API keys you need to get the constellation annotator working properly.

## 🔑 Required API Keys

### 1. **Astrometry.net API Key** ⭐ **REQUIRED**

**Purpose**: Plate solving - determines celestial coordinates of your astronomical images

**How to Get**:
1. Visit: https://nova.astrometry.net/
2. Click "Sign Up" or "Login"
3. Create a free account
4. Go to your profile page
5. Copy your API key

**Setup**:
```bash
# Set as environment variable
export ASTROMETRY_API_KEY="your_api_key_here"

# Or use with command line
python constellation_annotator.py input.jpg output.jpg --api-key YOUR_KEY
```

**Cost**: FREE (with rate limits)
- Free tier: 500 solves per day
- Paid tiers available for higher usage

**Why You Need This**: This is the core functionality - without it, the script cannot determine where your image is pointing in the sky.

---

## 🔧 Optional API Keys (For Future Enhancements)

### 2. **SIMBAD API Access** (Optional)

**Purpose**: Get additional star information and coordinates

**How to Get**:
1. Visit: http://simbad.u-strasbg.fr/simbad/
2. Register for an account
3. Request API access

**Cost**: FREE for academic/research use

**Note**: Currently not used in the basic version, but could be added for enhanced star catalogs.

### 3. **VizieR API Access** (Optional)

**Purpose**: Access to additional astronomical catalogs

**How to Get**:
1. Visit: http://vizier.u-strasbg.fr/
2. Register for an account
3. Request API access

**Cost**: FREE for academic/research use

**Note**: Currently not used in the basic version, but could be added for more comprehensive data.

---

## 🚀 Getting Started Checklist

### Step 1: Get Your Astrometry.net API Key
- [ ] Visit https://nova.astrometry.net/
- [ ] Create free account
- [ ] Copy API key from profile
- [ ] Set environment variable: `export ASTROMETRY_API_KEY="your_key"`

### Step 2: Test Your Setup
```bash
# Test that the key is set
echo $ASTROMETRY_API_KEY

# Test the constellation annotator
python test_demo.py
```

### Step 3: Process Your First Image
```bash
# Basic usage
python constellation_annotator.py your_image.jpg output.jpg

# With verbose output
python constellation_annotator.py your_image.jpg output.jpg --verbose
```

---

## 🔍 Troubleshooting API Issues

### Common Problems:

1. **"No API key provided"**
   - Solution: Set the environment variable or use `--api-key` flag

2. **"API key invalid"**
   - Solution: Check your key at https://nova.astrometry.net/
   - Make sure you copied the full key

3. **"Rate limit exceeded"**
   - Solution: Wait for rate limit to reset (daily)
   - Consider upgrading to paid tier for higher limits

4. **"Network error"**
   - Solution: Check your internet connection
   - Try again later

---

## 📊 API Usage Limits

### Astrometry.net Free Tier:
- **500 solves per day**
- **File size limit**: 50MB
- **Image formats**: JPEG, PNG, FITS
- **Rate limit**: ~1 request per 2 seconds

### Paid Tiers (if needed):
- **$5/month**: 5000 solves/day
- **$20/month**: 25000 solves/day
- **Custom**: Higher limits available

---

## 🛡️ Security Notes

### Best Practices:
1. **Never commit API keys to Git**
   - Use environment variables
   - Add `*.key` to `.gitignore`

2. **Keep keys private**
   - Don't share in public repositories
   - Use different keys for development/production

3. **Monitor usage**
   - Check your Astrometry.net dashboard
   - Watch for rate limit warnings

---

## 🔄 Environment Setup Examples

### For macOS/Linux:
```bash
# Add to your shell profile (~/.bashrc, ~/.zshrc, etc.)
export ASTROMETRY_API_KEY="your_api_key_here"

# Reload profile
source ~/.bashrc  # or ~/.zshrc
```

### For Windows:
```cmd
# Set environment variable
setx ASTROMETRY_API_KEY "your_api_key_here"

# Or set for current session
set ASTROMETRY_API_KEY=your_api_key_here
```

### For Python scripts:
```python
import os
os.environ['ASTROMETRY_API_KEY'] = 'your_api_key_here'
```

---

## 📞 Support

### If you have issues:
1. Check the troubleshooting section above
2. Visit Astrometry.net documentation
3. Check the Dinner Plate README.md
4. Open an issue on the GitHub repository

### Useful Links:
- **Astrometry.net**: https://nova.astrometry.net/
- **Dinner Plate Repository**: https://github.com/Charlietcooper/dinner-plate
- **Astrometry.net API Docs**: https://astrometry.net/doc/net/api.html

---

## 🎯 Summary

**You only need ONE API key to get started:**
- ✅ **Astrometry.net API Key** (Required)

**Optional for future enhancements:**
- 🔄 SIMBAD API Access
- 🔄 VizieR API Access

Get your Astrometry.net key and you're ready to start annotating your astronomical images! 🌟 