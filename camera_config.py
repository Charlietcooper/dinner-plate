#!/usr/bin/env python3
"""
Camera Configuration for Dinner Plate Constellation Annotator

This file contains camera and lens information to improve plate solving accuracy.
"""

# Default camera configuration
DEFAULT_CAMERA_CONFIG = {
    "camera": "Canon EOS 200D (Rebel SL2)",
    "lens": "EF-S 18-55mm f/4-5.6 IS STM",
    "sensor_width_mm": 22.3,  # APS-C sensor width
    "sensor_height_mm": 14.9,  # APS-C sensor height
    "focal_length_mm": 18,     # Default focal length (wide angle)
    "aperture": 4.0,           # Default aperture
    "crop_factor": 1.6,        # Canon APS-C crop factor
}

# Common camera configurations
CAMERA_CONFIGS = {
    "canon_200d": {
        "camera": "Canon EOS 200D (Rebel SL2)",
        "lens": "EF-S 18-55mm f/4-5.6 IS STM",
        "sensor_width_mm": 22.3,
        "sensor_height_mm": 14.9,
        "focal_length_mm": 18,
        "aperture": 4.0,
        "crop_factor": 1.6,
    },
    "canon_200d_55mm": {
        "camera": "Canon EOS 200D (Rebel SL2)",
        "lens": "EF-S 18-55mm f/4-5.6 IS STM",
        "sensor_width_mm": 22.3,
        "sensor_height_mm": 14.9,
        "focal_length_mm": 55,
        "aperture": 5.6,
        "crop_factor": 1.6,
    },
    "generic_dslr": {
        "camera": "Generic DSLR",
        "lens": "Generic Lens",
        "sensor_width_mm": 23.5,  # Standard APS-C
        "sensor_height_mm": 15.6,
        "focal_length_mm": 50,
        "aperture": 5.6,
        "crop_factor": 1.5,
    },
    "full_frame": {
        "camera": "Full Frame DSLR",
        "lens": "Generic Lens",
        "sensor_width_mm": 36.0,
        "sensor_height_mm": 24.0,
        "focal_length_mm": 50,
        "aperture": 5.6,
        "crop_factor": 1.0,
    },
}

def get_camera_config(camera_name: str = "canon_200d") -> dict:
    """Get camera configuration by name."""
    return CAMERA_CONFIGS.get(camera_name, DEFAULT_CAMERA_CONFIG)

def calculate_field_of_view(config: dict) -> tuple:
    """Calculate field of view in degrees."""
    import math
    
    # Calculate field of view using focal length and sensor size
    fov_horizontal = 2 * math.atan(config["sensor_width_mm"] / (2 * config["focal_length_mm"]))
    fov_vertical = 2 * math.atan(config["sensor_height_mm"] / (2 * config["focal_length_mm"]))
    
    # Convert to degrees
    fov_horizontal_deg = math.degrees(fov_horizontal)
    fov_vertical_deg = math.degrees(fov_vertical)
    
    return fov_horizontal_deg, fov_vertical_deg

def get_plate_solve_params(config: dict) -> dict:
    """Get parameters for plate solving based on camera config."""
    fov_h, fov_v = calculate_field_of_view(config)
    
    # Calculate scale in arcseconds per pixel (approximate)
    # This is a rough estimate based on field of view and typical image resolution
    scale_est = fov_h * 3600 / 4000  # Assuming ~4000 pixel width
    
    return {
        "scale_est": scale_est,  # Estimated scale in arcseconds per pixel
        "scale_lower": scale_est * 0.5,  # Lower bound
        "scale_upper": scale_est * 2.0,  # Upper bound
        "scale_units": "arcsecperpix",
        "center_ra": None,  # Will be determined by plate solving
        "center_dec": None,
        "radius": fov_h / 2,  # Search radius in degrees
    }

def print_camera_info(config: dict):
    """Print camera configuration information."""
    fov_h, fov_v = calculate_field_of_view(config)
    
    print(f"📷 Camera: {config['camera']}")
    print(f"🔍 Lens: {config['lens']}")
    print(f"📐 Focal Length: {config['focal_length_mm']}mm")
    print(f"📏 Sensor: {config['sensor_width_mm']}mm x {config['sensor_height_mm']}mm")
    print(f"🌍 Field of View: {fov_h:.1f}° x {fov_v:.1f}°")
    print(f"📊 Crop Factor: {config['crop_factor']}x")

if __name__ == "__main__":
    # Test the camera configuration
    print("🌟 Dinner Plate - Camera Configuration Test")
    print("=" * 50)
    
    config = get_camera_config("canon_200d")
    print_camera_info(config)
    
    print(f"\n🔧 Plate Solve Parameters:")
    params = get_plate_solve_params(config)
    for key, value in params.items():
        print(f"  {key}: {value}") 