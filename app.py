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
        * All paths: 5 km/h (3.1 mph), standard pedestrian speed

      - Turn Penalties: 5-second delay per edge approximating intersection costs
      - Time Calculation: (Network distance / speed) + turn penalty per edge
      - Network Buffer: Dynamically sized to cover all candidate and demand points

   B. Manhattan Distance (default):
      - Method: Rectilinear distance times a road circuity factor (1.2x)
      - Speed: Average 25 mph for driving, 5 km/h for walking

   NOTE: OpenStreetMap does not include historical traffic data. For ArcGIS-style
   historical speeds, you would need Esri StreetMap Premium or HERE data (commercial).
   This tool uses posted speed limits from OSM as the best free alternative.

4. COVERAGE DETERMINATION:
   - A facility "covers" a demand point if travel time <= threshold
   - Binary coverage matrix: C[i,j] = 1 if facility i covers demand point j, 0 otherwise
   - Coverage is weighted by uninsured population at each demand point

5. OPTIMIZATION SOLVER:
   - Solver: PuLP with CBC (COIN-OR Branch and Cut) solver
   - Problem Type: Integer Linear Programming (ILP)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Iterable, Optional, Tuple

import folium
import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pulp import LpMaximize, LpProblem, LpVariable, PULP_CBC_CMD, lpSum
from shapely.geometry import Polygon
from streamlit_folium import st_folium

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
# BRANDING AND CUSTOM CSS
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
def render_folium_map(m: folium.Map, key: str = "map", height: int = 640):
    try:
        st_folium(
            m,
            key=key,
            height=height,
            use_container_width=True,
            returned_objects=[],  # stops recursion in some Streamlit + folium combos
        )
    except Exception:
        components.html(m.get_root().render(), height=height, scrolling=True)

# ===========================
# CONFIGURATION
# ===========================
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
if "covered_mask" not in st.session_state:
    st.session_state.covered_mask = None
if "method_used" not in st.session_state:
    st.session_state.method_used = "Manhattan Distance"
if "last_params" not in st.session_state:
    st.session_state.last_params = {}
if "selected_cand_ids" not in st.session_state:
    st.session_state.selected_cand_ids = None
if "covered_dem_ids" not in st.session_state:
    st.session_state.covered_dem_ids = None

# ===========================
# DATA LOADING
# ===========================
@st.cache_data
def load_data(json_path: Path) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
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

    # Stable row identifiers for fast membership tests later
    candidates_df["cand_idx"] = np.arange(len(candidates_df), dtype=int)
    demand_df["dem_idx"] = np.arange(len(demand_df), dtype=int)

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
        candidates_gdf = gpd.GeoDataFrame(
            candidates_df,
            geometry=gpd.points_from_xy(candidates_df["longitude"], candidates_df["latitude"]),
            crs="EPSG:4326",
        )
    else:
        raise ValueError("Candidates must include latitude and longitude.")

    if "longitude" in demand_df.columns and "latitude" in demand_df.columns:
        demand_gdf = gpd.GeoDataFrame(
            demand_df,
            geometry=gpd.points_from_xy(demand_df["longitude"], demand_df["latitude"]),
            crs="EPSG:4326",
        )
    else:
        raise ValueError("Demand points must include latitude and longitude.")

    # Precompute ZIP membership once (faster than repeated intersects filters)
    try:
        _zip = zip_gdf[["ZIP_CODE", "geometry"]].copy()
        candidates_gdf = gpd.sjoin(candidates_gdf, _zip, how="left", predicate="intersects").drop(columns=["index_right"])
        demand_gdf = gpd.sjoin(demand_gdf, _zip, how="left", predicate="intersects").drop(columns=["index_right"])
        candidates_gdf = candidates_gdf.rename(columns={"ZIP_CODE": "zip_join"})
        demand_gdf = demand_gdf.rename(columns={"ZIP_CODE": "zip_join"})
    except Exception:
        # If sjoin fails for any reason, fall back to original zip_code column later
        if "zip_join" not in candidates_gdf.columns:
            candidates_gdf["zip_join"] = np.nan
        if "zip_join" not in demand_gdf.columns:
            demand_gdf["zip_join"] = np.nan

    return zip_gdf, candidates_gdf, demand_gdf

# ===========================
# NETWORK HELPERS
# ===========================
def nearest_node_safe(G: nx.MultiDiGraph, lon: float, lat: float, max_snap_dist_m: float = MAX_SNAP_DIST_M):
    try:
        node, dist = ox.distance.nearest_nodes(G, lon, lat, return_dist=True)
        if dist is not None and dist > max_snap_dist_m:
            return None
        return node
    except TypeError:
        try:
            return ox.distance.nearest_nodes(G, lon, lat)
        except Exception:
            return None
    except Exception:
        return None

def snap_points_to_nodes(
    G: nx.MultiDiGraph,
    lons: np.ndarray,
    lats: np.ndarray,
    max_snap_dist_m: float = MAX_SNAP_DIST_M,
) -> np.ndarray:
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    out = np.empty(len(lons), dtype=object)

    try:
        nodes, dists = ox.distance.nearest_nodes(G, X=lons, Y=lats, return_dist=True)
        nodes = np.asarray(nodes, dtype=object)
        dists = np.asarray(dists, dtype=float)
        nodes[dists > max_snap_dist_m] = None
        out[:] = nodes
        return out
    except Exception:
        for i, (lon, lat) in enumerate(zip(lons, lats)):
            out[i] = nearest_node_safe(G, float(lon), float(lat), max_snap_dist_m)
        return out

def estimate_required_graph_dist_m(
    center_lat: float,
    center_lon: float,
    candidates_df: Optional[pd.DataFrame],
    demand_df: Optional[pd.DataFrame],
    min_dist: int = 15000,
    buffer_m: int = 5000,
) -> int:
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
# SPEED AND TRAVEL TIME
# ===========================
def preprocess_network_speeds(G: nx.MultiDiGraph, network_type: str = "drive") -> nx.MultiDiGraph:
    if network_type == "walk":
        for _, _, _, data in G.edges(data=True, keys=True):
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
                    for _, _, _, data in G.edges(data=True, keys=True):
                        road_type = data.get("highway", "unclassified")
                        if isinstance(road_type, list):
                            road_type = road_type[0]
                        data["speed_kph"] = SC_HIGHWAY_SPEEDS_KMH.get(road_type, SC_FALLBACK_SPEED_KMH)
                        length_km = data["length"] / 1000.0
                        data["travel_time"] = (length_km / data["speed_kph"]) * 3600.0

        for _, _, _, data in G.edges(data=True, keys=True):
            tt_sec = data.get("travel_time", 0)
            data["travel_time"] = (tt_sec + TURN_PENALTY_SECONDS) / 60.0

    return G

@st.cache_resource(show_spinner=False)
def get_osm_graph(center_lat: float, center_lon: float, dist_m: int, network_type: str) -> nx.MultiDiGraph:
    G = ox.graph_from_point((center_lat, center_lon), dist=int(dist_m), network_type=network_type)
    G = preprocess_network_speeds(G, network_type)
    return G

def calculate_manhattan_distance_time(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    mode: str = "drive",
) -> float:
    lat_diff = abs(dest_lat - origin_lat) * 69.0
    lon_diff = abs(dest_lon - origin_lon) * 69.0 * np.cos(np.radians(origin_lat))
    manhattan_miles = (lat_diff + lon_diff) * CIRCUITY_FACTOR

    if mode == "drive":
        return (manhattan_miles / DEFAULT_DRIVING_SPEED) * 60.0

    manhattan_km = manhattan_miles * 1.60934
    return (manhattan_km / WALKING_SPEED_KMH) * 60.0

# ===========================
# COVERAGE MATRIX
# ===========================
def build_coverage_matrix(
    candidates_subset: gpd.GeoDataFrame,
    demand_subset: gpd.GeoDataFrame,
    max_time: float,
    network_type: str = "drive",
    use_network: bool = False,
    G: Optional[nx.MultiDiGraph] = None,
) -> Tuple[np.ndarray, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    candidates_reset = candidates_subset.reset_index(drop=True).copy()
    demand_reset = demand_subset.reset_index(drop=True).copy()

    n_facilities = len(candidates_reset)
    n_demand = len(demand_reset)

    if n_facilities == 0 or n_demand == 0:
        return np.zeros((n_facilities, n_demand), dtype=np.uint8), candidates_reset, demand_reset

    if use_network and G is not None:
        coverage = np.zeros((n_facilities, n_demand), dtype=np.uint8)

        dem_nodes = snap_points_to_nodes(
            G,
            demand_reset["longitude"].to_numpy(),
            demand_reset["latitude"].to_numpy(),
            MAX_SNAP_DIST_M,
        )
        fac_nodes = snap_points_to_nodes(
            G,
            candidates_reset["longitude"].to_numpy(),
            candidates_reset["latitude"].to_numpy(),
            MAX_SNAP_DIST_M,
        )

        # Dijkstra per facility, destinations are already snapped
        for i, origin_node in enumerate(fac_nodes):
            if origin_node is None:
                continue
            try:
                lengths = nx.single_source_dijkstra_path_length(
                    G, origin_node, cutoff=float(max_time), weight="travel_time"
                )
            except Exception:
                continue

            row = np.fromiter(
                ((lengths.get(n, np.inf) <= max_time) if n is not None else False for n in dem_nodes),
                dtype=np.bool_,
                count=n_demand,
            )
            coverage[i, :] = row.astype(np.uint8)

        return coverage, candidates_reset, demand_reset

    # Manhattan distance, fully vectorized
    clat = candidates_reset["latitude"].to_numpy(dtype=float)[:, None]
    clon = candidates_reset["longitude"].to_numpy(dtype=float)[:, None]
    dlat = demand_reset["latitude"].to_numpy(dtype=float)[None, :]
    dlon = demand_reset["longitude"].to_numpy(dtype=float)[None, :]

    lat_diff = np.abs(dlat - clat) * 69.0
    lon_diff = np.abs(dlon - clon) * 69.0 * np.cos(np.radians(clat))
    miles = (lat_diff + lon_diff) * CIRCUITY_FACTOR

    if network_type == "drive":
        tt = (miles / DEFAULT_DRIVING_SPEED) * 60.0
    else:
        km = miles * 1.60934
        tt = (km / WALKING_SPEED_KMH) * 60.0

    coverage = (tt <= float(max_time)).astype(np.uint8)
    return coverage, candidates_reset, demand_reset

# ===========================
# OPTIMIZATION SOLVER
# ===========================
def solve_maxcover(
    coverage_matrix: np.ndarray,
    demand_weights: np.ndarray,
    num_facilities: int,
) -> Tuple[list[int], float, np.ndarray]:
    n_facilities, n_demand = coverage_matrix.shape

    model = LpProblem("Max_Coverage", LpMaximize)
    x = LpVariable.dicts("facility", range(n_facilities), cat="Binary")
    y = LpVariable.dicts("covered", range(n_demand), cat="Binary")

    demand_weights = np.asarray(demand_weights, dtype=float)

    model += lpSum(demand_weights[j] * y[j] for j in range(n_demand))
    model += lpSum(x[i] for i in range(n_facilities)) == int(num_facilities)

    # Precompute coverers for each demand point (faster than scanning in every constraint)
    for j in range(n_demand):
        coverers = np.where(coverage_matrix[:, j] == 1)[0]
        if coverers.size:
            model += y[j] <= lpSum(x[int(i)] for i in coverers)
        else:
            model += y[j] == 0

    model.solve(PULP_CBC_CMD(msg=0))

    selected = [i for i in range(n_facilities) if x[i].varValue is not None and x[i].varValue > 0.5]

    if selected:
        covered_mask = coverage_matrix[selected, :].any(axis=0)
        covered_demand = float(demand_weights[covered_mask].sum())
    else:
        covered_mask = np.zeros(n_demand, dtype=bool)
        covered_demand = 0.0

    return selected, covered_demand, covered_mask

# ===========================
# MAP CREATION
# ===========================
def create_map(
    zip_gdf: gpd.GeoDataFrame,
    selected_zip: str,
    candidates_df: gpd.GeoDataFrame,
    demand_df: gpd.GeoDataFrame,
    selected_cand_ids: Optional[set[int]] = None,
    covered_dem_ids: Optional[set[int]] = None,
    tiles: str = "CartoDB positron",
) -> folium.Map:
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

    folium.GeoJson(
        zip_boundary.geometry.__geo_interface__,
        style_function=lambda _: {
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

    # Fast ZIP filters if zip_join exists
    if "zip_join" in candidates_df.columns and candidates_df["zip_join"].notna().any():
        candidates_in_zip = candidates_df[candidates_df["zip_join"] == selected_zip]
    else:
        candidates_in_zip = candidates_df[candidates_df.geometry.intersects(zip_boundary.geometry)]

    if "zip_join" in demand_df.columns and demand_df["zip_join"].notna().any():
        demand_in_zip = demand_df[demand_df["zip_join"] == selected_zip]
    else:
        demand_in_zip = demand_df[demand_df.geometry.intersects(zip_boundary.geometry)]

    analysis_complete = (selected_cand_ids is not None) and (covered_dem_ids is not None)

    # Facilities
    if analysis_complete:
        for _, facility in candidates_in_zip.iterrows():
            lat = float(facility["latitude"])
            lon = float(facility["longitude"])
            name = str(facility.get("name", ""))
            ftype = str(facility.get("type", ""))

            is_selected = int(facility.get("cand_idx", -1)) in selected_cand_ids

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
            uninsured = float(demand["uninsured_pop"]) if pd.notna(demand.get("uninsured_pop", np.nan)) else 0.0

            is_covered = int(demand.get("dem_idx", -1)) in covered_dem_ids

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
            uninsured = float(demand["uninsured_pop"]) if pd.notna(demand.get("uninsured_pop", np.nan)) else 0.0

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

    # Dynamic map fit: padding scales with ZIP size so small ZIPs don't look tiny
    minx, miny, maxx, maxy = map(float, bounds)
    width = max(maxx - minx, 1e-12)
    height = max(maxy - miny, 1e-12)

    pad_frac = 0.08  # 8% of the bbox size
    pad_x = min(max(width * pad_frac, 0.0015), 0.25)
    pad_y = min(max(height * pad_frac, 0.0015), 0.25)

    m.fit_bounds(
        [
            [miny - pad_y, minx - pad_x],
            [maxy + pad_y, maxx + pad_x],
        ]
    )
    return m

# ===========================
# MAIN APP
# ===========================
def main():
    st.title("🏥 South Carolina MHC Placement Decision Tool")
    st.markdown("**Optimizing healthcare accessibility for South Carolina’s uninsured communities.**")

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

            # candidates in ZIP and by types, for max number of sites
            if "zip_join" in candidates_df.columns and candidates_df["zip_join"].notna().any():
                candidates_zip_all = candidates_df[candidates_df["zip_join"] == selected_zip]
            else:
                candidates_zip_all = candidates_df[candidates_df.geometry.intersects(zip_geom)]

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
    if "zip_join" in demand_df.columns and demand_df["zip_join"].notna().any():
        demand_in_zip = demand_df[demand_df["zip_join"] == selected_zip]
    else:
        demand_in_zip = demand_df[demand_df.geometry.intersects(zip_geom)]

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
        st.session_state.coverage_matrix = None
        st.session_state.demand_reset = None
        st.session_state.candidates_reset = None
        st.session_state.covered_mask = None
        st.session_state.selected_cand_ids = None
        st.session_state.covered_dem_ids = None

    # LAYOUT
    col_map, col_insights = st.columns([7, 3], gap="large")

    with col_insights:
        st.subheader("📊 Summary Statistics")
        st.metric("Total Uninsured Population", f"{int(round(total_uninsured)):,}")
        st.metric("Available Candidate Sites", f"{len(candidates_in_zip):,}")
        st.metric("Demand Points", f"{len(demand_in_zip):,}")

    # If no candidates
    if len(candidates_in_zip) == 0:
        with col_map:
            st.warning("No candidate facilities available for this ZIP and selected facility types.")
            m = create_map(zip_gdf, selected_zip, candidates_df, demand_df, tiles=map_tiles)
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
                    graph_dist_m = estimate_required_graph_dist_m(
                        zip_center.y,
                        zip_center.x,
                        candidates_in_zip,
                        demand_in_zip,
                        min_dist=15000,
                        buffer_m=buffer_m,
                    )
                    with st.spinner("Downloading or loading cached road network..."):
                        G = get_osm_graph(zip_center.y, zip_center.x, int(graph_dist_m), network_type)
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

            demand_weights = demand_reset["uninsured_pop"].to_numpy(dtype=float)
            selected_indices, covered_pop, covered_mask = solve_maxcover(
                coverage_matrix,
                demand_weights,
                int(num_facilities),
            )

            selected_facilities = candidates_reset.iloc[selected_indices]

            # For fast map rendering, use stable IDs
            selected_cand_ids = set(selected_facilities["cand_idx"].astype(int).tolist()) if "cand_idx" in selected_facilities.columns else set()
            covered_dem_ids = set(demand_reset.loc[covered_mask, "dem_idx"].astype(int).tolist()) if "dem_idx" in demand_reset.columns else set()

            st.session_state.analysis_complete = True
            st.session_state.selected_facilities = selected_facilities
            st.session_state.coverage_matrix = coverage_matrix
            st.session_state.demand_reset = demand_reset
            st.session_state.candidates_reset = candidates_reset
            st.session_state.covered_pop = covered_pop
            st.session_state.covered_mask = covered_mask
            st.session_state.method_used = method_used
            st.session_state.last_params = current_params
            st.session_state.selected_cand_ids = selected_cand_ids
            st.session_state.covered_dem_ids = covered_dem_ids

    # Map display
    with col_map:
        st.subheader("🗺️ Map")

        if st.session_state.analysis_complete:
            m = create_map(
                zip_gdf,
                selected_zip,
                candidates_df,
                demand_df,
                selected_cand_ids=st.session_state.selected_cand_ids,
                covered_dem_ids=st.session_state.covered_dem_ids,
                tiles=map_tiles,
            )
        else:
            m = create_map(zip_gdf, selected_zip, candidates_df, demand_df, tiles=map_tiles)

        render_folium_map(
            m,
            key=f"map_{selected_zip}_{'done' if st.session_state.analysis_complete else 'base'}",
            height=640,
        )

    # Coverage metrics
    if st.session_state.analysis_complete:
        with col_insights:
            cov_pop = float(st.session_state.covered_pop)
            pct = (cov_pop / total_uninsured) * 100 if total_uninsured > 0 else 0.0

            covered_count = int(np.sum(st.session_state.covered_mask)) if st.session_state.covered_mask is not None else 0
            total_pts = len(st.session_state.demand_reset) if st.session_state.demand_reset is not None else 0

            st.metric("Covered Uninsured Population", f"{int(round(cov_pop)):,}")
            st.metric("Coverage Percentage", f"{pct:.1f}%")
            st.metric("Covered Demand Points", f"{covered_count} / {total_pts}")
            st.progress(min(max(pct / 100.0, 0.0), 1.0))
            st.caption(f"Method: {st.session_state.method_used}")

    # Results table + export
    if st.session_state.analysis_complete:
        st.divider()
        st.subheader("📍 Recommended Site Details")

        selected_facilities = st.session_state.selected_facilities
        coverage_matrix = st.session_state.coverage_matrix
        demand_reset = st.session_state.demand_reset
        covered_mask = st.session_state.covered_mask

        covered_demand_count = int(np.sum(covered_mask)) if covered_mask is not None else 0
        st.caption(f"Covered demand points: {covered_demand_count:,} of {len(demand_reset):,}")

        # Individual coverage per selected facility
        demand_w = demand_reset["uninsured_pop"].to_numpy(dtype=float)
        facility_coverage = []
        for idx, _facility in selected_facilities.iterrows():
            indiv_pop = float(demand_w[coverage_matrix[idx, :] == 1].sum())
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
