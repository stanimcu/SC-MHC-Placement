# Quick Start Guide

## Get Running in 5 Minutes

### 1. Prerequisites
- Python 3.9+ installed
- Your data file ready at the path specified in the requirements

### 2. Setup
[README.md](../Github/sc_location_tool/README.md)
```bash
# Clone or download this folder, then navigate to it
cd sc_location_tool

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (takes 5-10 minutes)
pip install -r requirements.txt
```

### 3. Configure Your Data Path

Edit `app.py` line 24:
```python
JSON_PATH = Path("YOUR/PATH/HERE/sc_app_data.json")
```

### 4. Run

```bash
streamlit run app.py
```

Your browser will open automatically to `http://localhost:8501`

### 5. Use the Tool

1. Select a ZIP code from dropdown
2. Choose facility types
3. Set travel mode (drive/walk) and time threshold
4. Click "Run Analysis"
5. View results and export if needed

## Common Issues

**Import Error**: Run `pip install -r requirements.txt` again

**Data Not Loading**: Check your JSON_PATH in app.py

**Network Error**: Use Manhattan distance mode (uncheck network analysis)

**Map Not Showing**: Refresh browser or check if port 8501 is blocked

## Data Format Checklist

Your JSON file must have:
- ✅ `zip_boundaries` array with GeoJSON features
- ✅ `candidate_facilities` array with lat/lon/type/name
- ✅ `demand_points` array with lat/lon/uninsured_pop

See README.md for detailed format requirements.

## Need Help?

1. Read full README.md
2. Check data format matches examples
3. Verify all dependencies installed
4. Try with network analysis disabled first

## Stopping the App

Press `Ctrl+C` in the terminal where Streamlit is running.
