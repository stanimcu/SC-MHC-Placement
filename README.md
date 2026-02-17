# South Carolina MHC Placement Decision Tool

A Python-based interactive web application for optimal facility location selection using Maximum Coverage Location-Allocation analysis. Built with Streamlit for geospatial analysis of healthcare facility placement in South Carolina.

## Overview

This tool helps identify optimal facility locations to maximize coverage of uninsured populations within specified travel times. It uses:
- **Maximum Coverage Location Problem (MCLP)** optimization
- **Network-based or Manhattan distance** travel time calculations
- **Interactive mapping** with Folium
- **Real geospatial data** for South Carolina ZIP codes, facilities, and demand points

## Features

- ✅ Interactive ZIP code selection with dropdown
- ✅ Multi-select facility types (Churches, Community Health Clinics, Food Banks, Homeless Services, Hospitals, Rural Primary Care)
- ✅ Travel mode selection (Driving or Walking)
- ✅ Configurable travel time thresholds (5–45 minutes)
- ✅ Adjustable number of facilities to select
- ✅ Network-based travel time analysis using OSMnx
- ✅ Manhattan distance approximation (faster alternative)
- ✅ Interactive map with zoom and pan
- ✅ Coverage statistics and metrics
- ✅ Export results as CSV or GeoJSON

## Requirements

- Python 3.9 or higher
- Data file: `sc_app_data.json` (place in the project root folder)

## Installation

### Step 1: Create a Virtual Environment (Recommended)

```bash
# Navigate to your project directory
cd /path/to/sc_location_tool

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

> **Note:** Installation may take 5–10 minutes due to geospatial libraries.

### Step 3: Configure Data Path

Open `config.py` and update `JSON_PATH` to point to your data file:

```python
JSON_PATH = Path("sc_app_data.json")  # relative path if file is in the project root
# or an absolute path, e.g.:
# JSON_PATH = Path("/your/full/path/to/sc_app_data.json")
```

If `JSON_PATH` is not found and `ALLOW_FILE_UPLOAD = True`, the app will prompt you to upload the file directly in the browser.

## Data Format Requirements

Your JSON file must contain three top-level sections:

### 1. ZIP Code Boundaries (`zips`)
```json
{
  "zips": {
    "29630": {
      "po_name": "Clemson",
      "coords": [[lat, lon], [lat, lon], ...]
    }
  }
}
```

### 2. Candidate Facilities (`facilities`)
```json
{
  "facilities": [
    {
      "facility_id": "F001",
      "type": "Churches",
      "name": "First Baptist Church",
      "address": "123 Main St",
      "latitude": 34.6834,
      "longitude": -82.8374,
      "zip_code": "29630"
    }
  ]
}
```

Valid `type` values: `Churches`, `Community Health Clinics`, `Food Banks`, `Homeless Services`, `Hospitals`, `Rural Primary Care`

### 3. Demand Points (`demand`)
```json
{
  "demand": [
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

```bash
# Activate your virtual environment first
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

# Run the app
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

You can also use the provided setup scripts:
- **macOS/Linux**: `bash setup.sh`
- **Windows**: `setup.bat`

## Usage Guide

### 1. Set Parameters (Sidebar)

| Control | Description |
|---|---|
| **Focus ZIP Code** | Select a ZIP code from the dropdown (includes city name) |
| **Eligible Site Types** | Choose which facility types to include |
| **Travel Mode** | Drive or Walk |
| **Max Travel Time** | Coverage threshold in minutes (5–45 min) |
| **Target Number of Sites** | How many facilities to optimally select |
| **Enable Road-Network Routing** | Toggle for OSM road network (accurate but slower) vs. Manhattan distance (fast) |

### 2. Run the Analysis

Click **"🚀 Calculate Optimal Sites"** to:
- Build a coverage matrix based on travel times
- Solve the Maximum Coverage optimization problem
- Display selected facilities on the map
- Show coverage statistics

### 3. View Results

**Interactive Map:**
- Blue boundary = Selected ZIP code
- Colored markers = Candidate facilities by type (purple = Churches, red = Community Health Clinics, green = Food Banks, blue = Homeless Services, dark red = Hospitals, pink = Rural Primary Care)
- Gold stars = Optimally selected facilities
- Yellow/green circles = Demand points (sized by uninsured population; green = covered, yellow = uncovered)

**Summary Statistics Panel:**
- Total uninsured population in ZIP
- Number of available candidate sites
- Covered uninsured population and percentage
- Covered vs. total demand points

### 4. Export Results

Download the selected facilities as:
- **CSV** — for spreadsheet analysis
- **GeoJSON** — for GIS software (QGIS, ArcGIS, etc.)

## Coverage Calculation Method

### Coverage Definition
A demand point (census block centroid) is "covered" if it falls within the specified travel time threshold of at least one selected facility. Total coverage = sum of uninsured population at all covered demand points.

### Travel Time Methods

#### Manhattan Distance (Default — Fast)
- Rectilinear distance multiplied by a road circuity factor of 1.20
- Driving speed: 25 mph average
- Walking speed: 5 km/h (3.1 mph)
- No internet connection required

#### Road Network Analysis (Optional — More Accurate)
- Road network downloaded from OpenStreetMap via OSMnx
- Speed data from actual OSM `maxspeed` tags, with SC-appropriate fallbacks per road class
- Routing via Dijkstra's shortest path with a 5-second turn penalty per edge
- Network is cached after the first download per ZIP code

### Optimization Model
**Maximum Coverage Location Problem (MCLP)**:
- **Objective**: Maximize total uninsured population covered within the travel time threshold
- **Constraint**: Select exactly the specified number of facilities
- **Solver**: PuLP with the CBC (COIN-OR Branch and Cut) integer linear programming solver

## Troubleshooting

| Problem | Solution |
|---|---|
| **Error loading data** | Check `JSON_PATH` in `config.py`; ensure the file is valid JSON with all required fields |
| **No candidate facilities available** | Try different facility types or a different ZIP code |
| **Network analysis is slow** | First download takes 1–2 min per ZIP; use Manhattan distance for faster results |
| **Import errors** | Re-run `pip install -r requirements.txt`; upgrade pip with `pip install --upgrade pip` |
| **Map not displaying / shaking** | Ensure you are using `streamlit-folium >= 0.15.0` with `returned_objects=[]` |
| **Build errors on macOS (M1/M2)** | Install geospatial dependencies via `conda` instead of `pip` |

## Technical Architecture

**Backend:** Streamlit · GeoPandas · PuLP (CBC solver) · OSMnx · NetworkX · Shapely

**Frontend:** Folium · streamlit-folium

**Data flow:**
1. Load `sc_app_data.json` → parse into GeoDataFrames (cached)
2. User sets parameters → filter candidates and demand points by ZIP and type
3. Calculate travel times → build binary coverage matrix
4. Solve MCLP optimization → select optimal facility set
5. Render interactive map + statistics
6. Export to CSV / GeoJSON

## Performance Notes

| Operation | Typical Time |
|---|---|
| Data loading (first run) | 1–2 seconds (cached thereafter) |
| Manhattan distance analysis | 1–5 seconds |
| Network analysis (first run per ZIP) | 30–120 seconds |
| Network analysis (cached) | 5–10 seconds |
| Map rendering | 1–2 seconds |

## File Structure

```
sc_location_tool/
├── app.py                # Main Streamlit application
├── config.py             # Configuration (data path, defaults, settings)
├── requirements.txt      # Python dependencies
├── packages.txt          # System-level dependencies for cloud deployment
├── sc_app_data.json      # Geospatial data file (ZIP boundaries, facilities, demand)
├── setup.sh              # Linux/macOS setup script
├── setup.bat             # Windows setup script
└── README.md             # This file
```

## Future Enhancements

- Multi-objective optimization (coverage + cost)
- Temporal analysis (time-of-day traffic patterns)
- Demographic stratification (coverage by age/income group)
- Scenario comparison across multiple ZIP codes
- Integration with real-time population and insurance data

## Acknowledgments

- OpenStreetMap contributors for road network data
- PuLP / CBC for optimization capabilities
- Streamlit team for the application framework

## Citation

If you use this tool in your research, please cite the associated paper:

**APA**
> Tanim, S. H., White, D., Witrick, B., & Rennert, L. (2026). Optimizing Mobile Health Clinic Placement via Geospatial Modeling. *medRxiv*, 2025-12.

**MLA**
> Tanim, Shakhawat H., et al. "Optimizing Mobile Health Clinic Placement via Geospatial Modeling." *medRxiv* (2026): 2025-12.

**Chicago**
> Tanim, Shakhawat H., David White, Brian Witrick, and Lior Rennert. "Optimizing Mobile Health Clinic Placement via Geospatial Modeling." *medRxiv* (2026): 2025-12.

**Harvard**
> Tanim, S.H., White, D., Witrick, B. and Rennert, L., 2026. Optimizing Mobile Health Clinic Placement via Geospatial Modeling. *medRxiv*, pp.2025-12.

**Vancouver**
> Tanim SH, White D, Witrick B, Rennert L. Optimizing Mobile Health Clinic Placement via Geospatial Modeling. *medRxiv*. 2026 Jan 2:2025-12.

## License

This tool is provided as-is for research and analysis purposes.
