"""
South Carolina MHC Placement Decision Tool
A Streamlit application for optimal facility location selection using Maximum Coverage Location-Allocation

METHODOLOGY DOCUMENTATION:
========================

1. LOCATION-ALLOCATION MODEL TYPE:
   - Maximum Coverage Location Problem (MCLP)
   - Objective: Maximize the total covered demand (uninsured population) subject to selecting 
     exactly P facilities from a set of candidate locations
   
2. NETWORK ANALYSIS DATA SOURCES:
   When "Use Road Network Analysis" is enabled:
   - Network Data: OpenStreetMap (OSM) via OSMnx library
   - Road Network Type: 
     * 'drive' mode: Drivable roads with speeds from actual OSM maxspeed tags
     * 'walk' mode: Walkable paths (includes sidewalks, pedestrian paths)
   
3. TRAVEL TIME CALCULATION:
   
   A. Network-Based (when enabled):
      - Method: Dijkstra's shortest path algorithm on actual road network
      - Distance Metric: Network distance (meters along roads)
      - Speed Data:
        DRIVING:
        * Primary source: OSM `maxspeed` tags (actual posted speed limits)
        * Imputation: Mean maxspeed of same highway type when tag is missing
        * Fallback: SC-appropriate defaults per highway classification
        
        WALKING:
        * All paths: 5 km/h (3.1 mph) - standard pedestrian speed
      
      - Turn Penalties: 5-second delay per edge approximating intersection costs
      - Time Calculation: (Network distance / speed) + turn penalty per edge
      - Network Buffer: Dynamically sized to cover all candidate/demand points
   
   B. Manhattan Distance (default):
      - Method: Rectilinear distance x road circuity factor (1.2x)
      - Speed: Average 25 mph for driving, 5 km/h for walking
   
   NOTE: OpenStreetMap does not include historical traffic data. For ArcGIS-style
   historical speeds, you would need Esri StreetMap Premium / HERE data (commercial).
   This tool uses actual posted speed limits from OSM as the best free alternative.
   
4. COVERAGE DETERMINATION:
   - A facility "covers" a demand point if travel time <= threshold
   - Binary coverage matrix: C[i,j] = 1 if facility i covers demand point j, 0 otherwise
   - Coverage is weighted by uninsured population at each demand point
   
5. OPTIMIZATION SOLVER:
   - Solver: PuLP with CBC (COIN-OR Branch and Cut) solver
   - Problem Type: Integer Linear Programming (ILP)
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import geopandas as gpd
import folium
import json
from pathlib import Path
import numpy as np
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, PULP_CBC_CMD
import osmnx as ox
import networkx as nx
from shapely.geometry import Polygon
from streamlit_folium import st_folium
import warnings
from config import JSON_PATH

warnings.filterwarnings("ignore")

# ===========================
# PAGE CONFIG (must be first Streamlit command)
# ===========================
st.set_page_config(
    page_title="SC MHC Placement Decision Tool",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================
# BRANDING & CUSTOM CSS
# ===========================
def local_css():
    st.markdown(
        """
        <style>
            /* Hide Streamlit chrome (best-effort, selectors can change across versions) */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            [data-testid="stToolbar"] {visibility: hidden;}

            /* App background */
            .main {background-color: #f8f9fa;}

            /* Primary button */
            .stButton>button {
                width: 100%;
                border-radius: 8px;
                height: 3em;
                background-color: #004b98;
                color: white;
                font-weight: 700;
                border: 0px;
            }

            /* Tags for multiselect */
            .stMultiSelect [data-baseweb="tag"] {
                background-color: #004b98 !important;
                border-radius: 20px !important;
                padding: 5px 10px !important;
                margin: 2px !important;
                color: white !important;
            }

            /* Metric card */
            .metric-card {
                background-color: white;
                padding: 16px 18px;
                border-radius: 12px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.06);
                border-left: 6px solid #004b98;
            }

            /* Instruction box */
            .instruction-box {
                background-color: #e1f5fe;
                padding: 14px 16px;
                border-radius: 10px;
                border: 1px solid #b3e5fc;
                margin-bottom: 14px;
            }

            /* Slightly larger metric values */
            [data-testid="stMetricValue"] {font-size: 22px !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )

local_css()

# ===========================
# SAFE MAP RENDERING (fixes JSON serialization error from st_folium)
# ===========================
def render_folium_map(m, key="map", height=640, width=1000):
    try:
        st_folium(m, key=key, height=height, use_container_width=True)  # ← remove width, add this
    except Exception:
        components.html(m.get_root().render(), height=height, scrolling=True)

# ===========================
# CONFIGURATION
# ===========================

#JSON_PATH = Path("sc_app_data.json")


DEFAULT_USE_NETWORK = False
DEFAULT_NUM_FACILITIES = 3

WALKING_SPEED_KMH = 5.0

SC_HIGHWAY_SPEEDS_KMH = {
    "motorway": 105,
    "motorway_link": 72,
    "trunk": 89,
    "trunk_link": 64,
    "primary": 72,
    "primary_link": 56,
    "secondary": 56,
    "secondary_link": 48,
    "tertiary": 48,
    "tertiary_link": 40,
    "residential": 40,
    "living_street": 24,
    "service": 24,
    "unclassified": 40,
    "road": 40,
}
SC_FALLBACK_SPEED_KMH = 40
TURN_PENALTY_SECONDS = 5
CIRCUITY_FACTOR = 1.20
DEFAULT_DRIVING_SPEED = 25  # mph
MAX_SNAP_DIST_M = 2000

# ===========================
# SESSION STATE
# ===========================
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
if "selected_facilities" not in st.session_state:
    st.session_state.selected_facilities = None
if "coverage_matrix" not in st.session_state:
    st.session_state.coverage_matrix = None
if "demand_reset" not in st.session_state:
    st.session_state.demand_reset = None
if "candidates_reset" not in st.session_state:
    st.session_state.candidates_reset = None
if "covered_pop" not in st.session_state:
    st.session_state.covered_pop = 0.0
if "method_used" not in st.session_state:
    st.session_state.method_used = "Manhattan Distance"
if "last_params" not in st.session_state:
    st.session_state.last_params = {}

# ===========================
# DATA LOADING
# ===========================
@st.cache_data
def load_data(json_path: Path):
    with open(json_path, "r") as f:
        data = json.load(f)

    zip_data = data.get("zip_boundaries", data.get("zips", {}))
    if not zip_data:
        raise ValueError("No ZIP boundaries found in JSON.")

    geometries = []
    properties = []

    if isinstance(zip_data, dict):
        for zip_code, zip_info in zip_data.items():
            if "coords" not in zip_info:
                continue
            try:
                coords = zip_info["coords"]
                coords_lonlat = [[pt[1], pt[0]] for pt in coords]  # [lon, lat]
                geom = Polygon(coords_lonlat)
                geometries.append(geom)
                properties.append(
                    {
                        "ZIP_CODE": str(zip_code).zfill(5),
                        "po_name": zip_info.get("po_name", str(zip_code)),
                    }
                )
            except Exception:
                continue
    else:
        raise ValueError("ZIP boundaries must be a dictionary keyed by ZIP codes.")

    if not geometries:
        raise ValueError("No valid ZIP geometries could be parsed.")

    zip_gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs="EPSG:4326")

    candidates_data = data.get("candidate_facilities", data.get("facilities", []))
    if not candidates_data:
        raise ValueError("No candidate facilities found in JSON.")
    candidates_df = pd.DataFrame(candidates_data)

    demand_data = data.get("demand_points", data.get("demand", []))
    if not demand_data:
        raise ValueError("No demand points found in JSON.")
    demand_df = pd.DataFrame(demand_data)

    # Clean types
    for df in (candidates_df, demand_df):
        if "zip_code" in df.columns:
            df["zip_code"] = df["zip_code"].astype(str).str.zfill(5)
        for col in ("latitude", "longitude"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    if "uninsured_pop" in demand_df.columns:
        demand_df["uninsured_pop"] = pd.to_numeric(demand_df["uninsured_pop"], errors="coerce").fillna(0)

    # GeoDataFrames
    if "longitude" in candidates_df.columns and "latitude" in candidates_df.columns:
        candidates_df = gpd.GeoDataFrame(
            candidates_df,
            geometry=gpd.points_from_xy(candidates_df["longitude"], candidates_df["latitude"]),
            crs="EPSG:4326",
        )

    if "longitude" in demand_df.columns and "latitude" in demand_df.columns:
        demand_df = gpd.GeoDataFrame(
            demand_df,
            geometry=gpd.points_from_xy(demand_df["longitude"], demand_df["latitude"]),
            crs="EPSG:4326",
        )

    return zip_gdf, candidates_df, demand_df

# ===========================
# NETWORK HELPERS
# ===========================
def nearest_node_safe(G, lon, lat, max_snap_dist_m=MAX_SNAP_DIST_M):
    try:
        node, dist = ox.nearest_nodes(G, lon, lat, return_dist=True)
        if dist is not None and dist > max_snap_dist_m:
            return None
        return node
    except TypeError:
        try:
            return ox.nearest_nodes(G, lon, lat)
        except Exception:
            return None
    except Exception:
        return None

def estimate_required_graph_dist_m(center_lat, center_lon, candidates_df, demand_df, min_dist=15000, buffer_m=5000):
    pts = []
    if candidates_df is not None and len(candidates_df) > 0:
        pts.append(candidates_df[["latitude", "longitude"]])
    if demand_df is not None and len(demand_df) > 0:
        pts.append(demand_df[["latitude", "longitude"]])

    if not pts:
        return int(min_dist)

    allpts = pd.concat(pts, ignore_index=True).dropna()
    if allpts.empty:
        return int(min_dist)

    try:
        d = ox.distance.great_circle_vec(
            center_lat, center_lon, allpts["latitude"].values, allpts["longitude"].values
        )
        maxd = float(np.nanmax(d))
    except Exception:
        lat1 = np.radians(center_lat)
        lon1 = np.radians(center_lon)
        lat2 = np.radians(allpts["latitude"].values.astype(float))
        lon2 = np.radians(allpts["longitude"].values.astype(float))
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        maxd = float(np.nanmax(6371000 * c))

    if not np.isfinite(maxd):
        return int(min_dist)

    return int(max(min_dist, maxd + buffer_m))

# ===========================
# SPEED / TRAVEL TIME
# ===========================
def preprocess_network_speeds(G, network_type="drive"):
    if network_type == "walk":
        for u, v, k, data in G.edges(data=True, keys=True):
            data["speed_kph"] = WALKING_SPEED_KMH
            length_km = data["length"] / 1000.0
            tt_seconds = (length_km / WALKING_SPEED_KMH) * 3600.0
            data["travel_time"] = (tt_seconds + TURN_PENALTY_SECONDS) / 60.0
    else:
        try:
            G = ox.add_edge_speeds(G, hwy_speeds=SC_HIGHWAY_SPEEDS_KMH, fallback=SC_FALLBACK_SPEED_KMH)
            G = ox.add_edge_travel_times(G)  # seconds
        except (AttributeError, TypeError):
            try:
                G = ox.routing.add_edge_speeds(G, hwy_speeds=SC_HIGHWAY_SPEEDS_KMH, fallback=SC_FALLBACK_SPEED_KMH)
                G = ox.routing.add_edge_travel_times(G)
            except (AttributeError, TypeError):
                try:
                    G = ox.speed.add_edge_speeds(G, hwy_speeds=SC_HIGHWAY_SPEEDS_KMH, fallback=SC_FALLBACK_SPEED_KMH)
                    G = ox.speed.add_edge_travel_times(G)
                except Exception:
                    for u, v, k, data in G.edges(data=True, keys=True):
                        road_type = data.get("highway", "unclassified")
                        if isinstance(road_type, list):
                            road_type = road_type[0]
                        data["speed_kph"] = SC_HIGHWAY_SPEEDS_KMH.get(road_type, SC_FALLBACK_SPEED_KMH)
                        length_km = data["length"] / 1000.0
                        data["travel_time"] = (length_km / data["speed_kph"]) * 3600.0

        for u, v, k, data in G.edges(data=True, keys=True):
            tt_sec = data.get("travel_time", 0)
            data["travel_time"] = (tt_sec + TURN_PENALTY_SECONDS) / 60.0

    return G

def calculate_network_travel_time_preprocessed(G, origin, destinations_df, max_time_minutes):
    if G is None:
        return np.full(len(destinations_df), np.nan)

    origin_node = nearest_node_safe(G, origin[1], origin[0])
    if origin_node is None:
        return np.full(len(destinations_df), np.nan)

    try:
        lengths = nx.single_source_dijkstra_path_length(G, origin_node, cutoff=max_time_minutes, weight="travel_time")
    except Exception:
        return np.full(len(destinations_df), np.nan)

    travel_times = []
    for _, dest in destinations_df.iterrows():
        dest_node = nearest_node_safe(G, dest["longitude"], dest["latitude"])
        if dest_node is None:
            travel_times.append(np.nan)
        elif dest_node in lengths:
            travel_times.append(lengths[dest_node])
        else:
            travel_times.append(np.nan)

    return np.array(travel_times)

def calculate_manhattan_distance_time(origin_lat, origin_lon, dest_lat, dest_lon, mode="drive"):
    lat_diff = abs(dest_lat - origin_lat) * 69
    lon_diff = abs(dest_lon - origin_lon) * 69 * np.cos(np.radians(origin_lat))
    manhattan_miles = (lat_diff + lon_diff) * CIRCUITY_FACTOR

    if mode == "drive":
        return (manhattan_miles / DEFAULT_DRIVING_SPEED) * 60

    manhattan_km = manhattan_miles * 1.60934
    return (manhattan_km / WALKING_SPEED_KMH) * 60

# ===========================
# COVERAGE MATRIX
# ===========================
def build_coverage_matrix(candidates_subset, demand_subset, max_time, network_type="drive", use_network=False, G=None):
    n_facilities = len(candidates_subset)
    n_demand = len(demand_subset)
    coverage = np.zeros((n_facilities, n_demand), dtype=int)

    candidates_reset = candidates_subset.reset_index(drop=True)
    demand_reset = demand_subset.reset_index(drop=True)

    if use_network and G is not None:
        G = preprocess_network_speeds(G, network_type)

    for i, facility in candidates_reset.iterrows():
        facility_point = (facility["latitude"], facility["longitude"])

        if use_network and G is not None:
            travel_times = calculate_network_travel_time_preprocessed(G, facility_point, demand_reset, max_time)
            for j in range(len(travel_times)):
                if not np.isnan(travel_times[j]) and travel_times[j] <= max_time:
                    coverage[i, j] = 1
        else:
            for j, demand_pt in demand_reset.iterrows():
                travel_time = calculate_manhattan_distance_time(
                    facility["latitude"],
                    facility["longitude"],
                    demand_pt["latitude"],
                    demand_pt["longitude"],
                    mode=network_type,
                )
                if travel_time <= max_time:
                    coverage[i, j] = 1

    return coverage, candidates_reset, demand_reset

# ===========================
# OPTIMIZATION SOLVER
# ===========================
def solve_maxcover(coverage_matrix, demand_weights, num_facilities):
    n_facilities, n_demand = coverage_matrix.shape

    model = LpProblem("Max_Coverage", LpMaximize)
    x = LpVariable.dicts("facility", range(n_facilities), cat="Binary")
    y = LpVariable.dicts("covered", range(n_demand), cat="Binary")

    model += lpSum([demand_weights[j] * y[j] for j in range(n_demand)])
    model += lpSum([x[i] for i in range(n_facilities)]) == num_facilities

    for j in range(n_demand):
        covering = [i for i in range(n_facilities) if coverage_matrix[i, j] == 1]
        if covering:
            model += y[j] <= lpSum([x[i] for i in covering])
        else:
            model += y[j] == 0

    model.solve(PULP_CBC_CMD(msg=0))

    selected = [i for i in range(n_facilities) if x[i].varValue is not None and x[i].varValue > 0.5]

    if selected:
        covered_mask = (coverage_matrix[selected, :].sum(axis=0) > 0)
        covered_demand = float(np.sum(np.asarray(demand_weights)[covered_mask]))
    else:
        covered_demand = 0.0

    return selected, covered_demand

# ===========================
# MAP CREATION
# ===========================
def create_map(
    zip_gdf,
    selected_zip,
    candidates_df,
    demand_df,
    selected_facilities=None,
    coverage_matrix=None,
    demand_reset=None,
    tiles="CartoDB positron",
):
    zip_boundary = zip_gdf[zip_gdf["ZIP_CODE"] == selected_zip].iloc[0]
    bounds = zip_boundary.geometry.bounds
    center_lat = float((bounds[1] + bounds[3]) / 2)
    center_lon = float((bounds[0] + bounds[2]) / 2)

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles=tiles,
        prefer_canvas=True,
    )

    # ZIP boundary
    folium.GeoJson(
        zip_boundary.geometry.__geo_interface__,
        style_function=lambda x: {
            "fillColor": "#E6F2FF",
            "color": "#1E90FF",
            "weight": 4,
            "fillOpacity": 0.08,
            "opacity": 0.9,
        },
    ).add_to(m)

    type_colors = {
        "Churches": "purple",
        "Community Health Clinics": "red",
        "Food Banks": "green",
        "Homeless Services": "blue",
        "Hospitals": "darkred",
        "Rural Primary Care": "pink",
    }

    # Spatial filter: candidates in ZIP
    if "geometry" in candidates_df.columns:
        candidates_in_zip = candidates_df[candidates_df.geometry.intersects(zip_boundary.geometry)]
    else:
        candidates_in_zip = candidates_df[candidates_df["zip_code"] == selected_zip]

    # Spatial filter: demand in ZIP
    if "geometry" in demand_df.columns:
        demand_in_zip = demand_df[demand_df.geometry.intersects(zip_boundary.geometry)]
    else:
        demand_in_zip = demand_df[demand_df["zip_code"] == selected_zip]

    analysis_complete = selected_facilities is not None and coverage_matrix is not None and demand_reset is not None

    # Compute covered demand indices (in demand_reset index-space)
    covered_demand_indices = set()
    if analysis_complete:
        selected_idx = list(selected_facilities.index)  # indices in candidates_reset space
        for j in range(len(demand_reset)):
            for i in selected_idx:
                if coverage_matrix[i, j] == 1:
                    covered_demand_indices.add(j)
                    break

    # Facilities
    if analysis_complete:
        # mark selected sites based on lat/lon matching
        for _, facility in candidates_in_zip.iterrows():
            lat = float(facility["latitude"])
            lon = float(facility["longitude"])
            name = str(facility.get("name", ""))
            ftype = str(facility.get("type", ""))

            is_selected = False
            for _, sel in selected_facilities.iterrows():
                if abs(lat - float(sel["latitude"])) < 0.0001 and abs(lon - float(sel["longitude"])) < 0.0001:
                    is_selected = True
                    break

            if is_selected:
                folium.Marker(
                    location=[lat, lon],
                    popup=f"<b>Proposed Site</b><br><b>{name}</b><br>{ftype}",
                    icon=folium.Icon(color="green", icon="star", prefix="fa"),
                ).add_to(m)
            else:
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=7,
                    popup=f"<b>{name}</b><br>{ftype}",
                    color="#4169E1",
                    fill=True,
                    fillColor="#6495ED",
                    fillOpacity=0.6,
                    weight=2,
                ).add_to(m)
    else:
        for _, facility in candidates_in_zip.iterrows():
            color = type_colors.get(str(facility.get("type", "")), "gray")
            folium.CircleMarker(
                location=[float(facility["latitude"]), float(facility["longitude"])],
                radius=5,
                popup=f"<b>{facility.get('name','')}</b><br>{facility.get('type','')}",
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.65,
                weight=2,
            ).add_to(m)

    # Demand points
    if analysis_complete:
        for _, demand in demand_in_zip.iterrows():
            lat = float(demand["latitude"])
            lon = float(demand["longitude"])
            uninsured = float(demand["uninsured_pop"]) if pd.notna(demand["uninsured_pop"]) else 0.0

            # find matching index in demand_reset (lat/lon match)
            demand_reset_index = None
            for reset_idx in range(len(demand_reset)):
                if (
                    abs(float(demand_reset.iloc[reset_idx]["latitude"]) - float(demand["latitude"])) < 0.0001
                    and abs(float(demand_reset.iloc[reset_idx]["longitude"]) - float(demand["longitude"])) < 0.0001
                ):
                    demand_reset_index = reset_idx
                    break

            is_covered = (demand_reset_index in covered_demand_indices)

            if is_covered:
                color = "#7CFC90"
                fill_color = "#7CFC90"
                popup_text = f"<b>Covered</b><br>Uninsured: {uninsured:,.0f}"
            else:
                color = "darkorange"
                fill_color = "orange"
                popup_text = f"<b>Uncovered</b><br>Uninsured: {uninsured:,.0f}"

            folium.CircleMarker(
                location=[lat, lon],
                radius=3,
                popup=popup_text,
                color=color,
                fill=True,
                fillColor=fill_color,
                fillOpacity=0.7,
                weight=1,
            ).add_to(m)
    else:
        for _, demand in demand_in_zip.iterrows():
            lat = float(demand["latitude"])
            lon = float(demand["longitude"])
            uninsured = float(demand["uninsured_pop"]) if pd.notna(demand["uninsured_pop"]) else 0.0

            folium.CircleMarker(
                location=[lat, lon],
                radius=3,
                popup=f"<b>Block Group Centroid</b><br>Uninsured: {uninsured:,.0f}",
                color="darkorange",
                fill=True,
                fillColor="orange",
                fillOpacity=0.7,
                weight=1,
            ).add_to(m)

    # Legend
    if analysis_complete:
        legend_html = """
        <div style="position: fixed;
                    bottom: 50px; right: 50px; width: 240px; height: auto;
                    background-color: white; z-index:9999; font-size:14px;
                    border:2px solid grey; border-radius: 8px; padding: 10px">
        <p style="margin:0; font-weight:bold; text-align:center;">Legend</p>
        <hr style="margin:6px 0;">
        <p style="margin:3px 0; font-weight:bold;">Facilities</p>
        <p style="margin:3px 0; margin-left:10px;"><i class="fa fa-star" style="color:green"></i> Proposed Site</p>
        <p style="margin:3px 0; margin-left:10px;"><span style="color:#4169E1; font-size:18px;">&#9679;</span> Other Sites</p>
        <hr style="margin:6px 0;">
        <p style="margin:3px 0; font-weight:bold;">Demand</p>
        <p style="margin:3px 0; margin-left:10px;"><span style="color:#7CFC90; font-size:14px;">&#9679;</span> Covered Blocks</p>
        <p style="margin:3px 0; margin-left:10px;"><span style="color:darkorange; font-size:14px;">&#9679;</span> Uncovered Blocks</p>
        </div>
        """
    else:
        legend_html = """
        <div style="position: fixed;
                    bottom: 50px; right: 50px; width: 240px; height: auto;
                    background-color: white; z-index:9999; font-size:14px;
                    border:2px solid grey; border-radius: 8px; padding: 10px">
        <p style="margin:0; font-weight:bold; text-align:center;">Legend</p>
        <hr style="margin:6px 0;">
        <p style="margin:3px 0; font-weight:bold;">Facility Types</p>
        <p style="margin:3px 0; margin-left:10px;"><span style="color:purple; font-size:20px;">&#9679;</span> Churches</p>
        <p style="margin:3px 0; margin-left:10px;"><span style="color:red; font-size:20px;">&#9679;</span> Community Health</p>
        <p style="margin:3px 0; margin-left:10px;"><span style="color:green; font-size:20px;">&#9679;</span> Food Banks</p>
        <p style="margin:3px 0; margin-left:10px;"><span style="color:blue; font-size:20px;">&#9679;</span> Homeless Services</p>
        <p style="margin:3px 0; margin-left:10px;"><span style="color:darkred; font-size:20px;">&#9679;</span> Hospitals</p>
        <p style="margin:3px 0; margin-left:10px;"><span style="color:pink; font-size:20px;">&#9679;</span> Rural Primary Care</p>
        <hr style="margin:6px 0;">
        <p style="margin:3px 0; font-weight:bold;">Demand</p>
        <p style="margin:3px 0; margin-left:10px;"><span style="color:darkorange; font-size:14px;">&#9679;</span> Block Group Centroid</p>
        </div>
        """

    m.get_root().html.add_child(folium.Element(legend_html))

    padding = 0.03
    m.fit_bounds(
        [
            [float(bounds[1]) - padding, float(bounds[0]) - padding],
            [float(bounds[3]) + padding, float(bounds[2]) + padding],
        ]
    )

    return m

# ===========================
# MAIN APP
# ===========================
def main():
    # HERO
    st.title("🏥 South Carolina MHC Placement Decision Tool")
    st.markdown(
        "**Optimizing healthcare accessibility for South Carolina’s uninsured communities.**"
    )

    st.markdown(
        """
        <div class="instruction-box">
            <b>How to use:</b>
            Pick a ZIP, choose site types, set travel mode and time, select how many sites to place,
            then click <b>Calculate Optimal Sites</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("📖 Methodology Documentation"):
        st.markdown(
            """
            **Model Type:** Maximum Coverage Location Problem (MCLP)

            **Network Analysis (optional):**
            - Data source: OpenStreetMap via OSMnx
            - Speeds: posted speed limits from OSM `maxspeed` tags 
            - Routing: Dijkstra shortest paths on a directed network
            - Turn penalties: 5 seconds per edge (approximation)

            **Manhattan Distance (default):**
            - Rectilinear distance times circuity factor (1.2)
            - Driving speed: 25 mph average, walking: 5 km/h    
            """
        )

    st.divider()

    # Load data
    try:
        with st.spinner("Loading geospatial data..."):
            zip_gdf, candidates_df, demand_df = load_data(JSON_PATH)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info(f"Check the data path: {JSON_PATH}")
        st.stop()

    # SIDEBAR
    with st.sidebar:
        st.header("Control Panel")

        with st.expander("📍 Geographic Scope", expanded=True):
            zip_options = sorted(zip_gdf["ZIP_CODE"].unique())
            zip_labels = [
                f"{zc} ({zip_gdf[zip_gdf['ZIP_CODE'] == zc].iloc[0]['po_name']})"
                for zc in zip_options
            ]
            zip_dict = dict(zip(zip_labels, zip_options))
            selected_zip_label = st.selectbox("Focus ZIP Code", options=zip_labels, index=0)
            selected_zip = zip_dict[selected_zip_label]

        zip_geom = zip_gdf[zip_gdf["ZIP_CODE"] == selected_zip].iloc[0].geometry

        with st.expander("⚙️ Model Constraints", expanded=True):
            facility_types = sorted(candidates_df["type"].unique())
            selected_types = st.multiselect("Eligible Site Types", options=facility_types, default=facility_types)

            travel_mode = st.radio("Travel Mode", options=["drive", "walk"], horizontal=True)
            default_time = 5 if travel_mode == "drive" else 10
            time_threshold = st.select_slider(
                "Max Travel Time (min)", options=[5, 10, 15, 20, 30, 45], value=default_time
            )

            # candidates in zip and by types, for max number of sites
            if "geometry" in candidates_df.columns:
                candidates_zip_all = candidates_df[candidates_df.geometry.intersects(zip_geom)]
            else:
                candidates_zip_all = candidates_df[candidates_df["zip_code"] == selected_zip]

            candidates_in_zip = candidates_zip_all[candidates_zip_all["type"].isin(selected_types)]
            max_facilities = len(candidates_in_zip)

            num_facilities = st.number_input(
                "Target Number of Sites",
                min_value=1,
                max_value=max(1, max_facilities),
                value=min(DEFAULT_NUM_FACILITIES, max(1, max_facilities)),
                disabled=(max_facilities == 0),
            )

        with st.expander("🗺️ Map Style", expanded=False):
            map_theme = st.radio("Theme", options=["Light", "Dark"], horizontal=True)
            map_tiles = "CartoDB positron" if map_theme == "Light" else "CartoDB dark_matter"

        with st.expander("🔬 Advanced Settings", expanded=True):
            use_network = st.toggle(
                "Enable Road-Network Routing",
                value=DEFAULT_USE_NETWORK,
                help="Uses OSM road network and routing, slower but more realistic than distance approximation.",
            )

        run_analysis = st.button("🚀 Calculate Optimal Sites", type="primary")

    # Demand in ZIP
    if "geometry" in demand_df.columns:
        demand_in_zip = demand_df[demand_df.geometry.intersects(zip_geom)]
    else:
        demand_in_zip = demand_df[demand_df["zip_code"] == selected_zip]

    total_uninsured = float(demand_in_zip["uninsured_pop"].sum()) if len(demand_in_zip) else 0.0

    current_params = {
        "zip": selected_zip,
        "types": tuple(selected_types),
        "mode": travel_mode,
        "time": time_threshold,
        "num": int(num_facilities),
        "network": bool(use_network),
        "tiles": map_tiles,
    }

    params_changed = (st.session_state.last_params != current_params)

    if params_changed and not run_analysis:
        st.session_state.analysis_complete = False
        st.session_state.selected_facilities = None

    # LAYOUT
    col_map, col_insights = st.columns([7, 3], gap="large")

    with col_insights:
        st.subheader("📊 Summary Statistics")

        st.metric("Total Uninsured Population", f"{int(round(total_uninsured)):,}")
        st.metric("Available Candidate Sites", f"{len(candidates_in_zip):,}")
        st.metric("Demand Points", f"{len(demand_in_zip):,}")

        #if not st.session_state.analysis_complete:
            #st.info("Set parameters, then click **Calculate Optimal Sites** to see coverage impact.")

    # If no candidates
    if len(candidates_in_zip) == 0:
        with col_map:
            st.warning("No candidate facilities available for this ZIP and selected facility types.")
            m = create_map(zip_gdf, selected_zip, candidates_df, demand_df, tiles=map_tiles)
            #render_folium_map(m, key=f"map_{selected_zip}_base", height=640, width=1000)
            render_folium_map(m, key=f"map_{selected_zip}_base", height=640)
        return

    # Run analysis
    if run_analysis:
        with st.spinner("Running optimization analysis..."):
            G = None
            method_used = "Manhattan Distance"

            if use_network:
                zip_center = zip_geom.centroid
                network_type = "drive" if travel_mode == "drive" else "walk"
                buffer_m = 5000 if travel_mode == "drive" else 2000

                try:
                    with st.spinner("Downloading road network..."):
                        graph_dist_m = estimate_required_graph_dist_m(
                            zip_center.y,
                            zip_center.x,
                            candidates_in_zip,
                            demand_in_zip,
                            min_dist=15000,
                            buffer_m=buffer_m,
                        )
                        G = ox.graph_from_point(
                            (zip_center.y, zip_center.x),
                            dist=graph_dist_m,
                            network_type=network_type,
                        )
                    method_used = "Road Network (OSM maxspeed tags, turn penalties)"
                except Exception as e:
                    st.warning(f"Network download failed, using Manhattan distance. Details: {e}")
                    G = None
                    method_used = "Manhattan Distance"

            coverage_matrix, candidates_reset, demand_reset = build_coverage_matrix(
                candidates_in_zip,
                demand_in_zip,
                time_threshold,
                network_type=travel_mode,
                use_network=use_network,
                G=G,
            )

            demand_weights = demand_reset["uninsured_pop"].values
            selected_indices, covered_pop = solve_maxcover(
                coverage_matrix,
                demand_weights,
                int(num_facilities),
            )

            selected_facilities = candidates_reset.iloc[selected_indices]

            st.session_state.analysis_complete = True
            st.session_state.selected_facilities = selected_facilities
            st.session_state.coverage_matrix = coverage_matrix
            st.session_state.demand_reset = demand_reset
            st.session_state.candidates_reset = candidates_reset
            st.session_state.covered_pop = covered_pop
            st.session_state.method_used = method_used
            st.session_state.last_params = current_params

    # Map display
    with col_map:
        st.subheader("🗺️ Map")

        if st.session_state.analysis_complete:
            m = create_map(
                zip_gdf,
                selected_zip,
                candidates_df,
                demand_df,
                selected_facilities=st.session_state.selected_facilities,
                coverage_matrix=st.session_state.coverage_matrix,
                demand_reset=st.session_state.demand_reset,
                tiles=map_tiles,
            )
        else:
            m = create_map(zip_gdf, selected_zip, candidates_df, demand_df, tiles=map_tiles)

        render_folium_map(m, key=f"map_{selected_zip}_{'done' if st.session_state.analysis_complete else 'base'}",
                          height=640)

    # Coverage metrics — rendered AFTER analysis so session state is populated
    if st.session_state.analysis_complete:
        with col_insights:
            cov_pop = float(st.session_state.covered_pop)
            pct = (cov_pop / total_uninsured) * 100 if total_uninsured > 0 else 0.0

            _cov_matrix = st.session_state.coverage_matrix
            _sel_fac = st.session_state.selected_facilities
            _dem_reset = st.session_state.demand_reset
            _sel_idx = list(_sel_fac.index)
            _covered_count = 0
            for _j in range(_cov_matrix.shape[1]):
                for _i in _sel_idx:
                    if _cov_matrix[_i, _j] == 1:
                        _covered_count += 1
                        break

            st.metric("Covered Uninsured Population", f"{int(round(cov_pop)):,}")
            st.metric("Coverage Percentage", f"{pct:.1f}%")
            st.metric("Covered Demand Points", f"{_covered_count} / {len(_dem_reset)}")
            st.progress(min(max(pct / 100.0, 0.0), 1.0))
            st.caption(f"Method: {st.session_state.method_used}")

    # Results table + export
    if st.session_state.analysis_complete:
        st.divider()
        st.subheader("📍 Recommended Site Details")

        selected_facilities = st.session_state.selected_facilities
        coverage_matrix = st.session_state.coverage_matrix
        demand_reset = st.session_state.demand_reset

        # Covered demand points count
        covered_demand_count = 0
        selected_idx = list(selected_facilities.index)  # indices align with candidates_reset positions
        for j in range(coverage_matrix.shape[1]):
            for i in selected_idx:
                if coverage_matrix[i, j] == 1:
                    covered_demand_count += 1
                    break

        st.caption(f"Covered demand points: {covered_demand_count:,} of {len(demand_reset):,}")

        # Individual coverage per selected facility (simple ranking)
        facility_coverage = []
        for idx, _facility in selected_facilities.iterrows():
            indiv_pop = float(np.sum(demand_reset["uninsured_pop"].values[coverage_matrix[idx, :] == 1]))
            facility_coverage.append(indiv_pop)

        df_display = selected_facilities.copy()
        df_display["Individual Coverage"] = [int(round(v)) for v in facility_coverage]
        df_display = df_display.sort_values("Individual Coverage", ascending=False).reset_index(drop=True)
        df_display.insert(0, "Rank", range(1, len(df_display) + 1))

        show_cols = ["Rank"]
        for c in ["name", "type", "address", "Individual Coverage"]:
            if c in df_display.columns:
                show_cols.append(c)

        st.dataframe(df_display[show_cols], use_container_width=True, hide_index=True)

        st.subheader("📥 Export Results")
        c1, c2 = st.columns(2)

        with c1:
            export_cols = ["facility_id", "name", "type", "address", "latitude", "longitude"]
            available_cols = [c for c in export_cols if c in selected_facilities.columns]
            csv_data = selected_facilities[available_cols].to_csv(index=False)
            st.download_button(
                label="Download Proposed Sites (CSV)",
                data=csv_data,
                file_name=f"proposed_sites_{selected_zip}.csv",
                mime="text/csv",
                key="csv_download",
            )

        with c2:
            gdf_selected = gpd.GeoDataFrame(
                selected_facilities,
                geometry=gpd.points_from_xy(selected_facilities["longitude"], selected_facilities["latitude"]),
                crs="EPSG:4326",
            )
            geojson_data = gdf_selected.to_json()
            st.download_button(
                label="Download Proposed Sites (GeoJSON)",
                data=geojson_data,
                file_name=f"proposed_sites_{selected_zip}.geojson",
                mime="application/geo+json",
                key="geojson_download",
            )

if __name__ == "__main__":
    main()
