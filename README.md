# South Carolina Facility Location Analysis Tool

A Python-based interactive web application for optimal facility location selection using Maximum Coverage Location-Allocation analysis. Built with Streamlit for geospatial analysis of healthcare facility placement in South Carolina.

## Overview

This tool helps identify optimal facility locations to maximize coverage of uninsured populations within specified travel times. It uses:
- **Maximum Coverage Location Problem (MCLP)** optimization
- **Network-based or Manhattan distance** travel time calculations
- **Interactive mapping** with Folium
- **Real geospatial data** for South Carolina ZIP codes, facilities, and demand points

## Features

- ✅ Interactive ZIP code selection with dropdown
- ✅ Multi-select facility types (Church, Primary Care, Grocery, etc.)
- ✅ Travel mode selection (Driving or Walking)
- ✅ Configurable travel time thresholds (5-30 minutes)
- ✅ Adjustable number of facilities to select
- ✅ Network-based travel time analysis using OSMnx
- ✅ Manhattan distance approximation (faster alternative)
- ✅ Interactive map with zoom and pan
- ✅ Coverage statistics and metrics
- ✅ Export results as CSV or GeoJSON

## Requirements

- Python 3.9 or higher
- Your data file at the specified JSON path

## Installation

### Step 1: Create a Virtual Environment (Recommended)

```bash
# Navigate to your project directory
cd /path/to/your/project

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** Installation may take 5-10 minutes due to geospatial libraries.

### Step 3: Configure Data Path

Open `app.py` and update line 24 with your data file path:

```python
JSON_PATH = Path("/Users/shtanim/Library/CloudStorage/Box-Box/Documents_Box_Tanim/MHC_Tool/data/raw/sc_app_data.json")
```

## Data Format Requirements

Your JSON file should contain three main sections:

### 1. ZIP Code Boundaries
```json
{
  "zip_boundaries": [
    {
      "type": "Feature",
      "properties": {
        "ZIP_CODE": "29630",
        "po_name": "Clemson"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[lon, lat], ...]]
      }
    }
  ]
}
```

### 2. Candidate Facilities
```json
{
  "candidate_facilities": [
    {
      "facility_id": "F001",
      "type": "Church",
      "name": "First Baptist Church",
      "address": "123 Main St",
      "latitude": 34.6834,
      "longitude": -82.8374,
      "zip_code": "29630"
    }
  ]
}
```

### 3. Demand Points (Census Block Centroids)
```json
{
  "demand_points": [
    {
      "demand_id": "D001",
      "uninsured_pop": 150,
      "latitude": 34.6850,
      "longitude": -82.8400,
      "zip_code": "29630"
    }
  ]
}
```

## Running the Application

### From Terminal (Recommended)

```bash
# Make sure your virtual environment is activated
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Run the Streamlit app
streamlit run app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`

### From VS Code

1. Open the `sc_location_tool` folder in VS Code
2. Select your Python interpreter (the one in your venv)
3. Open the integrated terminal (View → Terminal)
4. Run: `streamlit run app.py`

### From PyCharm

1. Open the `sc_location_tool` folder as a project
2. Configure the Python interpreter to use your virtual environment
3. Right-click `app.py` → Run with Python Console
4. Or use terminal: `streamlit run app.py`

**Note:** You do NOT need Jupyter Notebook for this application. It runs entirely through Streamlit's web interface.

## Usage Guide

### 1. Select Analysis Parameters (Sidebar)

- **ZIP Code**: Choose a ZIP code from the dropdown (includes post office name)
- **Facility Types**: Select which types to include (Church, Primary Care, Grocery, etc.)
- **Travel Mode**: Choose Driving or Walking
- **Travel Time**: Select maximum coverage time (5-30 minutes)
- **Number of Facilities**: Set how many facilities to select
- **Network Analysis**: Toggle for precise road network analysis (slower) vs. Manhattan distance (faster)

### 2. Run Analysis

Click the **"🔍 Run Analysis"** button to:
- Build coverage matrix based on travel times
- Solve the Maximum Coverage optimization problem
- Display selected facilities on the map
- Show coverage statistics

### 3. View Results

- **Interactive Map**: Pan, zoom, and click markers for details
  - Blue boundary = Selected ZIP code
  - Purple/Red/Green markers = Candidate facilities (by type)
  - Green stars = Selected optimal facilities
  - Yellow circles = Demand points (sized by uninsured population)
- **Statistics Panel**: View total and covered uninsured population
- **Selected Facilities**: Expand each to see details

### 4. Export Results

Download selected facilities as:
- **CSV**: For spreadsheet analysis
- **GeoJSON**: For GIS software

## Coverage Calculation Method

### Coverage Definition
Coverage is based on **uninsured population** within the specified travel time threshold:
- Each demand point (census block centroid) has an associated uninsured population count
- A demand point is "covered" if it's within the travel time threshold of at least one selected facility
- Total coverage = sum of uninsured populations at all covered demand points

### Travel Time Calculation Methods

#### 1. Manhattan Distance (Default - Faster)
- Approximates travel along a grid-like road network
- Distance = |Δlat| + |Δlon| (converted to miles)
- Travel time = distance / speed
  - Driving: 30 mph average
  - Walking: 3 mph average
- **Pros**: Fast, no external data required
- **Cons**: Less accurate for actual road routes

#### 2. Network Analysis (Optional - More Accurate)
- Uses actual road network data from OpenStreetMap via OSMnx
- Calculates shortest path along real roads
- Considers road network topology
- **Pros**: Most accurate travel times
- **Cons**: Slower, requires internet for first run (cached afterward)

### Optimization Model
Uses **Maximum Coverage Location Problem (MCLP)**:
- **Objective**: Maximize uninsured population covered within travel time threshold
- **Constraint**: Select exactly the specified number of facilities
- **Solver**: PuLP with CBC optimizer
- Guarantees optimal solution given the coverage matrix

## Troubleshooting

### "Error loading data"
- Verify your JSON file path in `app.py` line 24
- Ensure the file exists and is valid JSON
- Check that all required fields are present

### "No candidate facilities available"
- The selected ZIP code may not have facilities of the chosen types
- Try selecting different facility types or a different ZIP code

### Network analysis is slow
- First-time network download can take 1-2 minutes per ZIP code
- Results are cached for subsequent runs
- Consider using Manhattan distance for faster results

### Import errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Try upgrading pip: `pip install --upgrade pip`
- For macOS with M1/M2 chips, you may need to install packages via conda

### Map not displaying
- Check browser console for errors (F12)
- Try refreshing the page
- Ensure streamlit-folium is installed correctly

## Technical Architecture

### Backend Components
- **Streamlit**: Web application framework
- **GeoPandas**: Geospatial data manipulation
- **PuLP**: Linear programming optimization
- **OSMnx**: Road network analysis
- **NetworkX**: Graph algorithms

### Frontend Components
- **Folium**: Interactive mapping
- **Streamlit-Folium**: Integration bridge

### Data Flow
1. Load JSON data → Parse into DataFrames/GeoDataFrames
2. User selects parameters → Filter data
3. Calculate travel times → Build coverage matrix
4. Solve optimization → Select optimal facilities
5. Render results → Interactive map + statistics
6. Export options → CSV/GeoJSON download

## Performance Notes

- **Data Loading**: Cached after first load (~1-2 seconds)
- **Manhattan Distance**: ~1-5 seconds per analysis
- **Network Analysis**: ~30-120 seconds first time, ~5-10 seconds cached
- **Map Rendering**: ~1-2 seconds

## File Structure

```
sc_location_tool/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── venv/                 # Virtual environment (created by you)
```

## Future Enhancements

Potential features for future versions:
- Multi-objective optimization (coverage + cost)
- Temporal analysis (time-of-day traffic patterns)
- Demographic stratification (coverage by age/income groups)
- Scenario comparison tools
- Batch analysis across multiple ZIP codes

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review your data format against requirements
3. Check Streamlit documentation: https://docs.streamlit.io
4. Verify all dependencies are correctly installed

## License

This tool is provided as-is for research and analysis purposes.

## Acknowledgments

- OpenStreetMap contributors for road network data
- PuLP/CBC for optimization capabilities
- Streamlit team for the excellent framework
