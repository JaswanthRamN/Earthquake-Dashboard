# USGS Earthquake Dashboard

An interactive dashboard for visualizing and analyzing earthquake data from the USGS (United States Geological Survey) API.

## Features

- **Interactive Map Visualization**: View earthquakes on a global map with different visualization options (scatter, cluster, heatmap, 3D)
- **Time Series Analysis**: Analyze earthquake frequency over time (daily, hourly, weekly)
- **Regional Analysis**: Explore earthquake density by region
- **Statistical Analysis**: View key metrics and statistical summaries
- **Advanced Filtering**: Filter earthquakes by magnitude, depth, region, time, and more
- **Dark/Light Theme**: Toggle between dark and light themes for better visibility

## Screenshots

(Screenshots will be added here)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/earthquake-dashboard.git
   cd earthquake-dashboard
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the dashboard:
   ```
   streamlit run earthquake_dashboard.py
   ```

## Dependencies

- streamlit
- pandas
- plotly
- folium
- streamlit-folium
- numpy
- requests

## Data Source

This dashboard uses data from the [USGS Earthquake API](https://earthquake.usgs.gov/earthquakes/feed/v1.0/geojson.php).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- USGS for providing the earthquake data
- Streamlit for the amazing framework
- Plotly and Folium for the visualization libraries 