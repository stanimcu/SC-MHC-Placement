"""
Configuration file for SC Location Analysis Tool
Copy this file and modify the paths to match your data location
"""

from pathlib import Path

# ===========================
# DATA PATHS
# ===========================

# Path to your JSON data file
JSON_PATH = Path("sc_app_data.json")

# Allow user to upload file if JSON_PATH is not found?
ALLOW_FILE_UPLOAD = True

# Key mappings for your JSON structure
JSON_KEY_MAPPING = {
    'zip_boundaries': 'zips',
    'candidate_facilities': 'facilities',
    'demand_points': 'demand'
}

# ===========================
# ANALYSIS DEFAULTS
# ===========================

DEFAULT_TRAVEL_MODE = 'drive'
DEFAULT_TIME_THRESHOLD = 5
DEFAULT_NUM_FACILITIES = 3
DEFAULT_USE_NETWORK = False

# ===========================
# TRAVEL SPEED ASSUMPTIONS
# ===========================

DRIVING_SPEED_MPH = 25
WALKING_SPEED_MPH = 3

# ===========================
# MAP SETTINGS
# ===========================

DEFAULT_ZOOM = 11
MAP_TILES = 'OpenStreetMap'

# ===========================
# FACILITY TYPE COLORS
# ===========================

FACILITY_COLORS = {
    'Church': 'purple',
    'Primary Care': 'red',
    'Grocery': 'green',
    'Other': 'gray'
}

# ===========================
# OPTIMIZATION SETTINGS
# ===========================

MAX_OPTIMIZATION_TIME = 300
OPTIMIZATION_GAP = 0.0

# ===========================
# UI SETTINGS
# ===========================

TIME_OPTIONS = [5, 10, 15, 20, 30]
ENABLE_CACHING = True