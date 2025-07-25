#!/usr/bin/env python3
"""
Configuration file for Dinner Plate Constellation Annotator
"""

import os

# Astrometry.net API Key
ASTROMETRY_API_KEY = "kratfmjcbjnenufx"

# Set environment variable
os.environ['ASTROMETRY_API_KEY'] = ASTROMETRY_API_KEY

# Optional: Additional API keys for future enhancements
# SIMBAD_API_KEY = "your_simbad_key_here"
# VIZIER_API_KEY = "your_vizier_key_here"

# Configuration settings
DEFAULT_LINE_COLOR = (0, 255, 0)  # Green
DEFAULT_LINE_THICKNESS = 2
DEFAULT_STAR_COLOR = (255, 255, 0)  # Yellow
DEFAULT_STAR_RADIUS = 3
DEFAULT_TEXT_COLOR = (255, 255, 255)  # White
DEFAULT_FONT_SIZE = 16

# Plate solving settings
SOLVE_TIMEOUT = 120  # seconds
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB 