import streamlit as st
# Set page config with theme options - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    layout="wide",
    page_title="🌍 USGS Earthquake Dashboard",
    page_icon="🌍",
    initial_sidebar_state="expanded"
)

from seaborn import set_style
import pandas as pd
import plotly.express as px # type: ignore
import plotly.graph_objects as go # type: ignore # For more custom plots if needed
import numpy as np
from datetime import datetime, timedelta
import folium # type: ignore
from streamlit_folium import folium_static # type: ignore
from folium.plugins import MarkerCluster, HeatMap # type: ignore
import re
import requests
import json
import time
from io import StringIO

# --- Helper Function for Region Extraction (Copied from analysis script) ---
def extract_region(place):
    if not isinstance(place, str):
        return 'Unknown'
    parts = place.split(',')
    if len(parts) > 1:
        region_part = parts[-1].strip()
        # Common US states (add more if needed based on data)
        # Keep CA as abbreviation as it appears that way in data
        us_states = ["CA", "Alaska", "Oklahoma", "Oregon", "Nevada", "Hawaii", "Washington", "Texas", "Montana", "Idaho", "Utah", "Wyoming", "Arizona", "New Mexico", "Colorado", "Kansas", "Arkansas", "Missouri"]
        if region_part in us_states:
            return region_part
        else:
            # Assume it's a country or larger region name
            return region_part
    else:
        # Handle specific cases or return the original string if no comma
        place_lower = place.lower()
        # Prioritize known regions/countries that might appear without commas
        if 'tonga' in place_lower: return 'Tonga'
        if 'japan' in place_lower: return 'Japan'
        if 'new zealand' in place_lower: return 'New Zealand'
        if 'papua new guinea' in place_lower: return 'Papua New Guinea'
        if 'indonesia' in place_lower: return 'Indonesia'
        if 'chile' in place_lower: return 'Chile'
        if 'philippines' in place_lower: return 'Philippines'
        if 'mexico' in place_lower: return 'Mexico'
        if 'burma' in place_lower or 'myanmar' in place_lower: return 'Myanmar'
        if 'ridge' in place_lower: return place # Keep ocean ridges
        if 'islands' in place_lower and 'aleutian' not in place_lower: return place # Keep specific island groups (except Aleutians which usually have AK)

        # Default: return the original string if no specific rule matches
        return place
# ---------------------------------------------

# --- Load and Prepare Data --- 
# Cache the data loading to prevent reloading on every interaction
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_usgs_data():
    """Fetch earthquake data from USGS API"""
    try:
        # USGS API endpoint for earthquakes in the last 30 days
        url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.geojson"
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Parse the GeoJSON response
        data = response.json()
        
        # Extract features into a list of dictionaries
        features = data['features']
        earthquakes = []
        
        for feature in features:
            properties = feature['properties']
            geometry = feature['geometry']
            
            # Skip entries with missing or invalid magnitude
            if properties['mag'] is None:
                continue
                
            earthquake = {
                'time': pd.to_datetime(properties['time'], unit='ms'),
                'latitude': geometry['coordinates'][1],
                'longitude': geometry['coordinates'][0],
                'depth': geometry['coordinates'][2],
                'mag': float(properties['mag']),  # Ensure magnitude is a float
                'place': properties['place'],
                'type': properties['type'],
                'alert': properties.get('alert', None),
                'tsunami': properties.get('tsunami', 0),
                'felt': properties.get('felt', 0),
                'significance': properties.get('significance', 0)
            }
            earthquakes.append(earthquake)
        
        # Convert to DataFrame
        df = pd.DataFrame(earthquakes)
        
        # Extract region
        df['region'] = df['place'].apply(extract_region)
        
        # Create a column for visualization size (absolute magnitude)
        df['viz_size'] = df['mag'].abs()
        
        return df
    except Exception as e:
        st.error(f"Error fetching data from USGS API: {e}")
        return pd.DataFrame()

@st.cache_data
def load_data():
    try:
        # Try to fetch from API first
        df = fetch_usgs_data()
        
        # If API fetch fails, try to load from local file
        if df.empty:
            df = pd.read_csv('earthquakes_last_30_days.csv')
            # Convert time
            df['time'] = pd.to_datetime(df['time'])
            # Extract region
            df['region'] = df['place'].apply(extract_region)
            # Ensure latitude/longitude are numeric
            df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
            df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
            df = df.dropna(subset=['latitude', 'longitude'])
            
            # Create a column for visualization size (absolute magnitude)
            df['viz_size'] = df['mag'].abs()
        
        return df
    except FileNotFoundError:
        st.error("Error: The file 'earthquakes_last_30_days.csv' was not found.")
        st.info("Please ensure the file is in the same directory as the dashboard script.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred during data loading: {e}")
        return pd.DataFrame()

# Initialize session state for data refresh
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()

df = load_data()

# --- Dashboard Layout ---
# Theme toggle
theme = st.sidebar.radio("🎨 Select Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("""
        <style>
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        .stMetric {
            background-color: #262730;
            padding: 10px;
            border-radius: 5px;
        }
        .info-box {
            background-color: #262730;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .stApp {
            background-color: #FFFFFF;
            color: #31333F;
        }
        .stMetric {
            background-color: #F0F2F6;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        }
        .info-box {
            background-color: #F0F2F6;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 4px;
            border: none;
            padding: 8px 16px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        h1, h2, h3 {
            color: #1E88E5;
            font-weight: 600;
        }
        .stDataFrame {
            border-radius: 5px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        }
        </style>
    """, unsafe_allow_html=True)

# Main title and introduction
st.title('🌍 USGS Earthquake Dashboard')
st.markdown("""
<div class='info-box'>
    <h3>📊 About This Dashboard</h3>
    <p>This interactive dashboard displays real-time earthquake data from the USGS (United States Geological Survey). 
    You can explore earthquakes from the last 30 days with various visualization options and filters.</p>
    
    <h4>🔍 Key Features:</h4>
    <ul>
        <li>Interactive global map with multiple view options</li>
        <li>Time series analysis of earthquake frequency</li>
        <li>Regional analysis and statistics</li>
        <li>Customizable filters for detailed exploration</li>
        <li>Real-time data updates</li>
    </ul>
    
    <h4>💡 How to Use:</h4>
    <ol>
        <li>Use the sidebar filters to customize your view</li>
        <li>Switch between different map visualizations</li>
        <li>Explore the various analysis tabs</li>
        <li>Download filtered data for further analysis</li>
    </ol>
</div>
""", unsafe_allow_html=True)

# Add a header image with better styling
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/Seismogram.svg/1200px-Seismogram.svg.png", 
         caption="Earthquake Seismogram - Visual representation of seismic waves", use_column_width=True)

if not df.empty:
    # Add data refresh button and last refresh time with better styling
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.session_state.last_refresh = datetime.now()
            st.rerun()
    
    with col2:
        st.markdown(f"<p style='color: #666;'>Last updated: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}</p>", 
                   unsafe_allow_html=True)
    
    st.markdown("Visualizing recent earthquake data from USGS.")

    # --- Sidebar Filters ---
    st.sidebar.header("🔍 Filters")
    
    # Reset filters button with better styling
    if st.sidebar.button("🔄 Reset All Filters"):
        st.rerun()

    # Search box for locations with better placeholder
    search_query = st.sidebar.text_input("🔎 Search Location", 
                                       placeholder="Enter city, country, or region...")
    if search_query:
        # Case-insensitive search in place column
        search_results = df[df['place'].str.lower().str.contains(search_query.lower(), na=False)]
        if not search_results.empty:
            st.sidebar.write(f"Found {len(search_results)} matches")
            # Show matching locations
            for _, row in search_results.head(5).iterrows():
                st.sidebar.write(f"- {row['place']} (Mag: {row['mag']:.1f})")
        else:
            st.sidebar.write("No matches found")

    # Advanced filtering options with better organization
    st.sidebar.subheader("📅 Time Filters")
    
    # Time Range Filter with better labels
    min_time = df['time'].min()
    max_time = df['time'].max()
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=min_time.date(),
            min_value=min_time.date(),
            max_value=max_time.date(),
            help="Select the start date for the data range"
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=max_time.date(),
            min_value=min_time.date(),
            max_value=max_time.date(),
            help="Select the end date for the data range"
        )
    
    # Time of day filter
    time_of_day = st.sidebar.multiselect(
        "Time of Day",
        options=["Morning (6-12)", "Afternoon (12-18)", "Evening (18-24)", "Night (0-6)"],
        default=["Morning (6-12)", "Afternoon (12-18)", "Evening (18-24)", "Night (0-6)"]
    )
    
    # Convert time of day selections to hour ranges
    time_ranges = []
    for selection in time_of_day:
        if "Morning" in selection:
            time_ranges.append((6, 12))
        if "Afternoon" in selection:
            time_ranges.append((12, 18))
        if "Evening" in selection:
            time_ranges.append((18, 24))
        if "Night" in selection:
            time_ranges.append((0, 6))

    # Magnitude Slider
    min_mag, max_mag = float(df['mag'].min()), float(df['mag'].max())
    mag_range = st.sidebar.slider("Select Magnitude Range", min_value=min_mag, max_value=max_mag, value=(min_mag, max_mag))

    # Depth Slider
    min_depth, max_depth = float(df['depth'].min()), float(df['depth'].max())
    # Add a small buffer to max_depth if min and max are the same
    if min_depth == max_depth:
        max_depth += 1
    depth_range = st.sidebar.slider("Select Depth Range (km)", min_value=min_depth, max_value=max_depth, value=(min_depth, max_depth))

    # Region Multi-select
    all_regions = sorted(df['region'].unique())
    selected_regions = st.sidebar.multiselect("Select Regions", options=all_regions, default=all_regions)
    
    # Map Style Selector with better organization
    st.sidebar.subheader("🗺️ Map Settings")
    map_style = st.sidebar.selectbox(
        "Select Map Style",
        ["open-street-map", "satellite", "dark", "light", "terrain"],
        help="Choose the visual style of the map"
    )
    
    # Map View Options with better labels
    map_view = st.sidebar.radio(
        "Map View Type",
        ["Scatter", "Cluster", "Heatmap", "3D"],
        help="Choose how to display the earthquakes on the map"
    )
    
    # Additional filters
    st.sidebar.subheader("Additional Filters")
    
    # Filter by earthquake type
    all_types = sorted(df['type'].unique())
    selected_types = st.sidebar.multiselect("Earthquake Types", options=all_types, default=all_types)
    
    # Filter by significance
    min_significance = int(df['significance'].min())
    max_significance = int(df['significance'].max())
    significance_threshold = st.sidebar.slider(
        "Minimum Significance",
        min_value=min_significance,
        max_value=max_significance,
        value=0
    )
    
    # Filter by tsunami warning
    tsunami_filter = st.sidebar.checkbox("Show only tsunami warnings", value=False)
    
    # Filter by felt reports
    felt_filter = st.sidebar.checkbox("Show only felt earthquakes", value=False)

    # Significant Earthquake Alert Threshold
    significant_mag_threshold = st.sidebar.slider(
        "Significant Earthquake Alert Threshold",
        min_value=5.0,
        max_value=8.0,
        value=6.0,
        step=0.5
    )

    # Filter the DataFrame based on selections
    filtered_df = df[
        (df['time'].dt.date >= start_date) &
        (df['time'].dt.date <= end_date) &
        (df['mag'] >= mag_range[0]) &
        (df['mag'] <= mag_range[1]) &
        (df['depth'] >= depth_range[0]) &
        (df['depth'] <= depth_range[1]) &
        (df['region'].isin(selected_regions)) &
        (df['type'].isin(selected_types)) &
        (df['significance'] >= significance_threshold)
    ].copy()  # Create a copy to avoid SettingWithCopyWarning
    
    # Update viz_size after filtering
    filtered_df['viz_size'] = filtered_df['mag'].abs()
    
    # Apply time of day filter
    if time_ranges:
        time_filter = False
        for start_hour, end_hour in time_ranges:
            if start_hour < end_hour:
                time_filter = time_filter | ((filtered_df['time'].dt.hour >= start_hour) & (filtered_df['time'].dt.hour < end_hour))
            else:  # Handle overnight ranges (e.g., 22-6)
                time_filter = time_filter | ((filtered_df['time'].dt.hour >= start_hour) | (filtered_df['time'].dt.hour < end_hour))
        filtered_df = filtered_df[time_filter]
    
    # Apply tsunami filter
    if tsunami_filter:
        filtered_df = filtered_df[filtered_df['tsunami'] == 1]
    
    # Apply felt filter
    if felt_filter:
        filtered_df = filtered_df[filtered_df['felt'] > 0]

    st.sidebar.metric("Filtered Earthquakes", len(filtered_df))
    
    # Export visualizations
    st.sidebar.subheader("Export Options")
    if st.sidebar.button("Export Dashboard as Image"):
        st.sidebar.info("This feature requires additional setup. Please use your browser's screenshot functionality.")
    # ---------------------

    # --- Add dashboard elements below ---
    st.header("Global Earthquake Map")

    # Create tabs for different sections with better organization
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🗺️ Map View", "⏰ Time Analysis", "🌍 Regional Analysis", "📊 Statistics", "📋 Raw Data"])
    
    with tab1:
        st.markdown("""
        <div class='info-box'>
            <h4>🗺️ Map Visualization Options:</h4>
            <ul>
                <li><b>Scatter:</b> Individual points for each earthquake</li>
                <li><b>Cluster:</b> Grouped points for better visibility in dense areas</li>
                <li><b>Heatmap:</b> Intensity map showing earthquake density</li>
                <li><b>3D:</b> Three-dimensional view including depth</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Create map based on selected view
        if map_view == "Scatter":
            # Create the Plotly Express map with enhanced features
            fig_map = px.scatter_mapbox(
                filtered_df,
                lat="latitude",
                lon="longitude",
                size="viz_size",
                color="depth",
                hover_name="place",
                hover_data={
                    "mag": ":.2f",
                    "depth": ":.1f",
                    "time": True,
                    "latitude": False,
                    "longitude": False,
                    "viz_size": False
                },
                color_continuous_scale=px.colors.sequential.Viridis_r,
                size_max=15,  # Refined max point size for clarity
                zoom=1,
                mapbox_style=map_style,
                title="Earthquakes (Size=Magnitude, Color=Depth)"
            )
            
            # Enhance map layout
            fig_map.update_layout(
                margin={"r":0,"t":50,"l":0,"b":0},
                mapbox=dict(
                    center=dict(lat=0, lon=0),
                    zoom=1,
                    style=map_style
                ),
                hovermode='closest'
            )
            
            st.plotly_chart(fig_map, use_container_width=True)
            
            # Add legend and explanation
            st.markdown("""
            <div class='info-box'>
                <h4>📊 Map Legend:</h4>
                <ul>
                    <li><b>Point Size:</b> Represents earthquake magnitude</li>
                    <li><b>Color:</b> Represents depth (darker = deeper)</li>
                    <li><b>Hover:</b> Click on points for detailed information</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        elif map_view == "Cluster":
            # Create a Folium map with marker clusters
            m = folium.Map(location=[0, 0], zoom_start=2, tiles=map_style)
            marker_cluster = MarkerCluster().add_to(m)
            
            for _, row in filtered_df.iterrows():
                # Use absolute magnitude for marker size
                marker_size = min(20, max(5, abs(row['mag']) * 5))
                
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup=f"<b>{row['place']}</b><br>Magnitude: {row['mag']:.2f}<br>Depth: {row['depth']:.1f} km<br>Time: {row['time']}",
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(marker_cluster)
            
            folium_static(m)
        
        elif map_view == "Heatmap":
            # Create a Folium map with heatmap
            m = folium.Map(location=[0, 0], zoom_start=2, tiles=map_style)
            
            # Prepare heatmap data using absolute magnitude
            heat_data = [[row['latitude'], row['longitude'], abs(row['mag'])] for _, row in filtered_df.iterrows()]
            HeatMap(heat_data).add_to(m)
            
            folium_static(m)
        
        else:  # 3D view
            # Create a 3D scatter plot using Plotly
            fig_3d = px.scatter_3d(
                filtered_df,
                x="longitude",
                y="latitude",
                z="depth",
                size="viz_size",
                color="mag",
                hover_name="place",
                hover_data={
                    "mag": ":.2f",
                    "depth": ":.1f",
                    "time": True,
                    "longitude": False,
                    "latitude": False,
                    "viz_size": False
                },
                title="3D Earthquake Visualization (X=Longitude, Y=Latitude, Z=Depth)",
                labels={"longitude": "Longitude", "latitude": "Latitude", "depth": "Depth (km)", "mag": "Magnitude"}
            )
            
            fig_3d.update_layout(
                scene=dict(
                    zaxis=dict(
                        title="Depth (km)",
                        range=[max_depth, min_depth]  # Invert depth axis for better visualization
                    )
                )
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)

    # Significant Earthquakes Alert
    significant_quakes = filtered_df[filtered_df['mag'] >= significant_mag_threshold]
    if not significant_quakes.empty:
        st.warning(f"⚠️ **Significant Earthquakes Alert:** {len(significant_quakes)} earthquakes with magnitude ≥ {significant_mag_threshold}")
        for _, quake in significant_quakes.iterrows():
            st.write(f"- **{quake['place']}** (Mag: {quake['mag']:.1f}, Depth: {quake['depth']:.1f} km, Time: {quake['time']})")

    with tab2:
        st.markdown("""
        <div class='info-box'>
            <h4>⏰ Time Analysis Features:</h4>
            <ul>
                <li>Daily, hourly, and weekly earthquake frequency</li>
                <li>Trend analysis and patterns</li>
                <li>Time-based filtering options</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Time series options with better organization
        time_series_option = st.radio(
            "Select Time Scale",
            ["Daily", "Hourly", "Weekly"],
            horizontal=True,
            help="Choose the time scale for the analysis"
        )
        
        if time_series_option == "Daily":
            time_group = filtered_df.groupby(filtered_df['time'].dt.date).size().reset_index(name='count')
            time_group.columns = ['time', 'count']
            title = "Daily Earthquake Frequency"
        elif time_series_option == "Hourly":
            time_group = filtered_df.groupby(filtered_df['time'].dt.hour).size().reset_index(name='count')
            time_group.columns = ['time', 'count']
            title = "Hourly Earthquake Frequency"
        else:  # Weekly
            time_group = filtered_df.groupby(filtered_df['time'].dt.isocalendar().week).size().reset_index(name='count')
            time_group.columns = ['time', 'count']
            title = "Weekly Earthquake Frequency"
        
        fig_time = px.line(
            time_group,
            x='time',
            y='count',
            title=title,
            labels={'time': 'Date' if time_series_option == "Daily" else 'Hour' if time_series_option == "Hourly" else 'Week', 'count': 'Number of Earthquakes'}
        )
        st.plotly_chart(fig_time, use_container_width=True)
        
        # Add trend analysis
        st.subheader("Trend Analysis")
        
        # Calculate daily trend
        daily_trend = filtered_df.groupby(filtered_df['time'].dt.date).size().reset_index(name='count')
        daily_trend['date'] = pd.to_datetime(daily_trend['time'])
        
        # Simple linear regression for trend
        if len(daily_trend) > 1:
            x = np.arange(len(daily_trend))
            y = daily_trend['count'].values
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            
            # Calculate trend direction and strength
            trend_direction = "increasing" if z[0] > 0 else "decreasing"
            trend_strength = abs(z[0])
            
            # Display trend information
            st.write(f"The earthquake frequency is **{trend_direction}** with a strength of {trend_strength:.4f} events per day.")
            
            # Plot trend
            fig_trend = px.scatter(
                daily_trend,
                x='date',
                y='count',
                title="Daily Earthquake Count with Trend Line"
            )
            
            # Add trend line
            fig_trend.add_scatter(
                x=daily_trend['date'],
                y=p(np.arange(len(daily_trend))),
                mode='lines',
                name='Trend'
            )
            
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Not enough data points for trend analysis.")

    with tab3:
        st.markdown("""
        <div class='info-box'>
            <h4>🌍 Regional Analysis Features:</h4>
            <ul>
                <li>Geographic distribution of earthquakes</li>
                <li>Regional statistics and comparisons</li>
                <li>Depth and magnitude patterns by region</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Add region heatmap
        st.header("Earthquake Density by Region")
        region_counts = filtered_df['region'].value_counts().reset_index()
        region_counts.columns = ['region', 'count']
        
        fig_region_heatmap = px.choropleth(
            region_counts,
            locations='region',
            locationmode='country names',
            color='count',
            title="Earthquake Frequency by Region",
            color_continuous_scale=px.colors.sequential.Plasma
        )
        st.plotly_chart(fig_region_heatmap, use_container_width=True)
        
        # Add histograms for magnitude and depth based on filtered data
        st.subheader("Filtered Data Distributions")
        if not filtered_df.empty:
            col_hist1, col_hist2 = st.columns(2)

            with col_hist1:
                fig_mag_hist = px.histogram(
                    filtered_df, 
                    x="mag", 
                    nbins=30, 
                    title="Magnitude Distribution",
                    labels={'mag': 'Magnitude', 'count': 'Frequency'}
                )
                fig_mag_hist.update_layout(bargap=0.1)
                st.plotly_chart(fig_mag_hist, use_container_width=True)

            with col_hist2:
                fig_depth_hist = px.histogram(
                    filtered_df, 
                    x="depth", 
                    nbins=30, 
                    title="Depth Distribution (km)",
                    labels={'depth': 'Depth (km)', 'count': 'Frequency'}
                )
                fig_depth_hist.update_layout(bargap=0.1)
                st.plotly_chart(fig_depth_hist, use_container_width=True)
                
            # Add correlation plot
            st.subheader("Magnitude vs Depth Correlation")
            fig_corr = px.scatter(
                filtered_df,
                x="mag",
                y="depth",
                title="Magnitude vs Depth Correlation",
                labels={'mag': 'Magnitude', 'depth': 'Depth (km)'},
                trendline="ols"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Add type distribution
            st.subheader("Earthquake Types")
            type_counts = filtered_df['type'].value_counts().reset_index()
            type_counts.columns = ['type', 'count']
            
            fig_type = px.pie(
                type_counts,
                values='count',
                names='type',
                title="Distribution of Earthquake Types"
            )
            st.plotly_chart(fig_type, use_container_width=True)
        else:
            st.info("No data to display distributions for the current filter settings.")

    with tab4:
        st.markdown("""
        <div class='info-box'>
            <h4>📊 Statistical Analysis:</h4>
            <ul>
                <li>Summary statistics for selected data</li>
                <li>Magnitude and depth distributions</li>
                <li>Correlation analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.header("Data Overview")
        # Display some key metrics based on filtered data
        st.subheader("Filtered Data Metrics")
        if not filtered_df.empty:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Earthquakes", len(filtered_df))
            col2.metric("Max Magnitude", f"{filtered_df['mag'].max():.2f}")
            col3.metric("Average Magnitude", f"{filtered_df['mag'].mean():.2f}")
            col4.metric("Most Recent", filtered_df['time'].max().strftime("%Y-%m-%d %H:%M"))
            
            # Add statistical summary
            st.subheader("Statistical Summary")
            stats_df = filtered_df[['mag', 'depth']].describe()
            st.dataframe(stats_df, use_container_width=True)
            
            # Add advanced statistics
            st.subheader("Advanced Statistics")
            # Calculate additional statistics
            stats = {
                'Total Energy Released (estimated)': np.sum(10 ** (1.5 * filtered_df['mag'])),
                'Most Active Region': filtered_df['region'].mode().iloc[0],
                'Average Depth': filtered_df['depth'].mean(),
                'Depth Range': f"{filtered_df['depth'].min():.1f} - {filtered_df['depth'].max():.1f} km",
                'Magnitude Standard Deviation': filtered_df['mag'].std()
            }
            
            # Display statistics in a nice format
            for key, value in stats.items():
                st.metric(key, f"{value:.2f}" if isinstance(value, float) else value)
        else:
            st.info("No earthquakes match the current filter settings.")

    with tab5:
        st.markdown("""
        <div class='info-box'>
            <h4>📋 Raw Data Features:</h4>
            <ul>
                <li>Complete dataset with all details</li>
                <li>Sortable and filterable columns</li>
                <li>Export options for further analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Display the filtered data in a table with pagination
        st.header("Filtered Earthquake Data")
        # Select and rename columns for better readability
        display_columns = {
            'time': 'Time',
            'mag': 'Magnitude',
            'depth': 'Depth (km)',
            'place': 'Place',
            'region': 'Region',
            'type': 'Type',
            'significance': 'Significance',
            'tsunami': 'Tsunami Warning',
            'felt': 'Felt Reports'
        }
        
        # Add pagination
        page_size = 10
        total_pages = len(filtered_df) // page_size + (1 if len(filtered_df) % page_size > 0 else 0)
        page = st.number_input('Page', min_value=1, max_value=total_pages, value=1)
        
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        st.dataframe(
            filtered_df[display_columns.keys()].rename(columns=display_columns).iloc[start_idx:end_idx],
            use_container_width=True
        )

else:
    st.warning("Could not load earthquake data to build the dashboard.") 

# Add a footer with additional information
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <p>Data source: <a href='https://earthquake.usgs.gov/earthquakes/feed/v1.0/geojson.php' target='_blank'>USGS Earthquake API</a></p>
    <p>Last updated: {}</p>
    <p>Created with ❤️ using Streamlit</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d")), unsafe_allow_html=True) 