import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Added for better plots

# --- Helper Function for Region Extraction ---
def extract_region(place):
    if not isinstance(place, str):
        return 'Unknown'
    parts = place.split(',')
    if len(parts) > 1:
        region_part = parts[-1].strip()
        # Common US states (add more if needed based on data)
        us_states = ["CA", "AK", "OK", "OR", "NV", "HI", "WA", "TX", "MT", "ID", "UT", "WY", "AZ", "NM", "CO", "KS", "AR", "MO"]
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

# Load the dataset
try:
    # Assuming the CSV is in the same directory as the script
    df = pd.read_csv('earthquakes_last_30_days.csv')

    # --- 1. Time Conversion ---
    # Convert the 'time' column to datetime objects
    df['time'] = pd.to_datetime(df['time'])
    print("--- Converted 'time' column to datetime ---")

    # Display basic information
    print("First 5 rows of the dataset:")
    print(df.head())
    print("\n" + "="*50 + "\n") # Separator

    print("Dataset information (columns, types, non-null counts):")
    df.info()
    print("\n" + "="*50 + "\n") # Separator

    print("Summary statistics for numerical columns:")
    print(df.describe())
    print("\n" + "="*50 + "\n") # Separator

    # --- 2. Magnitude Distribution ---
    print("--- Generating Magnitude Distribution Histogram ---")
    plt.figure(figsize=(10, 6))
    plt.hist(df['mag'].dropna(), bins=30, edgecolor='black') # dropna() to handle potential NaNs
    plt.title('Distribution of Earthquake Magnitudes (Last 30 Days)')
    plt.xlabel('Magnitude')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()
    print("Histogram displayed. Close the plot window to continue.")
    print("\n" + "="*50 + "\n") # Separator

    # --- 3. Depth Distribution ---
    print("--- Generating Depth Distribution Histogram ---")
    plt.figure(figsize=(10, 6))
    # We might have negative depths (above sea level/errors?), let's see the distribution.
    plt.hist(df['depth'].dropna(), bins=50, edgecolor='black')
    plt.title('Distribution of Earthquake Depths (Last 30 Days)')
    plt.xlabel('Depth (km)')
    plt.ylabel('Frequency')
    # Log scale can be useful if depths are heavily skewed
    # plt.yscale('log')
    plt.grid(axis='y', alpha=0.75)
    plt.show()
    print("Histogram displayed. Close the plot window to continue.")
    print("\n" + "="*50 + "\n") # Separator

    # --- 4. Geographical Distribution (Simple Scatter Plot) ---
    print("--- Generating Geographical Distribution Scatter Plot ---")
    plt.figure(figsize=(12, 8))
    plt.scatter(df['longitude'], df['latitude'], alpha=0.5, s=df['mag']**2, label='Earthquakes') # Size by magnitude
    plt.title('Geographical Distribution of Earthquakes (Last 30 Days)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.legend()
    plt.show()
    print("Scatter plot displayed. Close the plot window to continue.")
    print("\n" + "="*50 + "\n") # Separator

    # --- 5. Time Series (Earthquakes per Day) ---
    print("--- Generating Time Series Plot (Earthquakes per Day) ---")
    # Set time as index for resampling
    df_time_indexed = df.set_index('time')
    # Resample by day ('D') and count occurrences
    earthquakes_per_day = df_time_indexed.resample('D').size()

    plt.figure(figsize=(12, 6))
    earthquakes_per_day.plot()
    plt.title('Number of Earthquakes per Day (Last 30 Days)')
    plt.xlabel('Date')
    plt.ylabel('Number of Earthquakes')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print("Time series plot displayed. Close the plot window to continue.")
    print("\n" + "="*50 + "\n") # Separator

    # --- 6. Largest Earthquakes ---
    print("--- Finding Top 10 Largest Earthquakes (by Magnitude) ---")
    top_10_earthquakes = df.nlargest(10, 'mag')

    print("Top 10 Largest Earthquakes in the Last 30 Days:")
    # Select and display relevant columns
    print(top_10_earthquakes[['time', 'mag', 'depth', 'place']])
    print("\n" + "="*50 + "\n") # Separator

    # --- 7. Investigate Magnitude Peaks ---
    print("--- Investigating Magnitude Peaks ---")
    # Define magnitude ranges for the peaks
    peak1_df = df[(df['mag'] >= 0.5) & (df['mag'] < 2.5)]
    peak2_df = df[df['mag'] >= 4.0]

    print(f"\nAnalysis of Low-Magnitude Peak (0.5 <= mag < 2.5): {len(peak1_df)} events")
    if not peak1_df.empty:
        print("Depth Statistics:")
        print(peak1_df['depth'].describe())
        print("\nMost Common Magnitude Types:")
        print(peak1_df['magType'].value_counts().head())
    else:
        print("No events found in this range.")

    print(f"\nAnalysis of Higher-Magnitude Peak (mag >= 4.0): {len(peak2_df)} events")
    if not peak2_df.empty:
        print("Depth Statistics:")
        print(peak2_df['depth'].describe())
        print("\nMost Common Magnitude Types:")
        print(peak2_df['magType'].value_counts().head())
    else:
        print("No events found in this range.")
    print("\n" + "="*50 + "\n") # Separator

    # --- 8. Focus on Stronger Earthquakes (mag >= 4.0) ---
    print("--- Generating Geographical Plot for Stronger Earthquakes (mag >= 4.0) ---")
    if not peak2_df.empty:
        plt.figure(figsize=(12, 8))
        plt.scatter(peak2_df['longitude'], peak2_df['latitude'], alpha=0.7, c=peak2_df['depth'], cmap='viridis', s=peak2_df['mag']**2.5) # Color by depth, size by magnitude
        plt.colorbar(label='Depth (km)')
        plt.title('Geographical Distribution of Stronger Earthquakes (mag >= 4.0)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True)
        plt.show()
        print("Scatter plot for stronger earthquakes displayed. Close the plot window to continue.")
    else:
        print("No earthquakes found with magnitude >= 4.0 to plot.")
    print("\n" + "="*50 + "\n") # Separator

    # --- 9. Magnitude vs. Depth ---
    print("--- Generating Magnitude vs. Depth Scatter Plot ---")
    plt.figure(figsize=(10, 7))
    plt.scatter(df['depth'], df['mag'], alpha=0.5, c=df['mag'], cmap='plasma', s=df['mag']**2)
    plt.colorbar(label='Magnitude')
    plt.title('Earthquake Magnitude vs. Depth (Last 30 Days)')
    plt.xlabel('Depth (km)')
    plt.ylabel('Magnitude')
    # plt.xscale('log') # Optional: Use log scale for depth if distribution is skewed
    plt.grid(True, alpha=0.5)
    plt.show()
    print("Magnitude vs. Depth plot displayed. Close the plot window to continue.")
    print("\n" + "="*50 + "\n") # Separator

    # --- 10. Analyze by Location (Top Locations) ---
    print("--- Identifying Top 10 Most Frequent Earthquake Locations ---")
    # Use regex to extract the general area if needed, or just use the raw 'place' string
    # For simplicity, let's use the raw place string for now
    top_locations = df['place'].value_counts().head(10)

    print("Top 10 most frequent locations reported:")
    print(top_locations)
    print("\nNote: 'Place' names can be specific. Further grouping might be needed for regional analysis.")
    print("\n" + "="*50 + "\n") # Separator

    # --- 11. Regional Analysis ---
    print("--- Performing Regional Analysis ---")
    # Apply the function to create the region column
    df['region'] = df['place'].apply(extract_region)

    # 11a. Calculate earthquake counts per region
    print("\n--- Earthquake Counts per Region (Top 15) ---")
    region_counts = df['region'].value_counts()
    print(region_counts.head(15))
    print("\n" + "-"*30 + "\n")

    # 11b. Analyze top regions (e.g., top 5 by count)
    print("--- Average Magnitude and Depth for Top 5 Regions ---")
    top_5_regions = region_counts.head(5).index.tolist()
    for region in top_5_regions:
        region_df = df[df['region'] == region]
        avg_mag = region_df['mag'].mean()
        avg_depth = region_df['depth'].mean()
        print(f"Region: {region} ({len(region_df)} events)")
        print(f"  Avg Magnitude: {avg_mag:.2f}")
        print(f"  Avg Depth: {avg_depth:.2f} km")
    print("\n" + "="*50 + "\n") # Separator

    # Plot magnitude distribution for specific regions (e.g., CA)
    print("--- Generating Magnitude Distribution for CA ---")
    ca_df = df[df['region'] == 'CA']
    if not ca_df.empty:
        plt.figure(figsize=(10, 6))
        plt.hist(ca_df['mag'].dropna(), bins=30, edgecolor='black')
        plt.title('Distribution of Earthquake Magnitudes in CA (Last 30 Days)')
        plt.xlabel('Magnitude')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.show()
        print("CA Magnitude plot displayed. Close the plot window to continue.")
    else:
        print("No data found for CA.")
    print("\n" + "="*50 + "\n") # Separator

    # --- 12. Deep Dive into Magnitude Types (magType) ---
    print("--- Analyzing Magnitude Types (magType) ---")

    # 12a. List unique magTypes and counts
    print("\n--- Unique Magnitude Types and Counts ---")
    magtype_counts = df['magType'].value_counts()
    print(magtype_counts)
    print("\n" + "-"*30 + "\n")

    # 12b. Plot Magnitude Distribution per magType (using Seaborn Boxplot)
    print("--- Generating Magnitude Distribution per magType (Boxplot) ---")
    plt.figure(figsize=(12, 7))
    # Only plot for types with a reasonable number of occurrences (e.g., >10)
    common_magtypes = magtype_counts[magtype_counts > 10].index
    sns.boxplot(data=df[df['magType'].isin(common_magtypes)], x='magType', y='mag', order=common_magtypes)
    plt.title('Magnitude Distribution by Magnitude Type')
    plt.xlabel('Magnitude Type')
    plt.ylabel('Magnitude')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    print("Magnitude Type boxplot displayed. Close the plot window to continue.")
    print("\n" + "-"*30 + "\n")

    # 12c. magType usage in Top 5 Regions
    print("--- Top Magnitude Types Used in Top 5 Regions ---")
    for region in top_5_regions: # top_5_regions calculated in step 11b
        region_df = df[df['region'] == region]
        print(f"Region: {region}")
        print(region_df['magType'].value_counts().head(3)) # Show top 3 types per region
        print("--")
    print("\n" + "="*50 + "\n") # Separator

    # --- 13. Compare Plate Boundary vs. Intraplate Regions ---
    print("--- Comparing Depth Distributions: Plate Boundary vs. Intraplate ---")

    # Define example regions for each type (USING FULL NAMES MATCHING df['region'])
    plate_boundary_regions = ['CA', 'Alaska', 'Indonesia', 'Puerto Rico', 'Papua New Guinea', 'Japan', 'Tonga', 'Chile', 'Philippines', 'Mexico', 'Myanmar', 'New Zealand'] # CA is abbreviation
    intraplate_regions = ['Texas', 'Oklahoma', 'Montana', 'Nevada', 'New Mexico', 'Washington', 'Oregon', 'Utah', 'Idaho', 'Hawaii'] # Hawaii is intraplate hotspot

    # Filter data
    boundary_df = df[df['region'].isin(plate_boundary_regions)]
    intraplate_df = df[df['region'].isin(intraplate_regions)]

    # Plot overlapping histograms of depth
    if not boundary_df.empty and not intraplate_df.empty:
        plt.figure(figsize=(12, 7))
        plt.hist(boundary_df['depth'].dropna(), bins=50, alpha=0.7, label='Plate Boundary', density=True)
        plt.hist(intraplate_df['depth'].dropna(), bins=50, alpha=0.7, label='Intraplate', density=True)
        plt.title('Depth Distribution: Plate Boundary vs. Intraplate Regions')
        plt.xlabel('Depth (km)')
        plt.ylabel('Density')
        plt.yscale('log') # Use log scale due to large differences in frequency
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.show()
        print("Depth comparison histogram displayed. Close the plot window to continue.")
    else:
        print("Could not plot comparison - one or both region groups are empty.")
    print("\n" + "="*50 + "\n") # Separator

    # --- 14. Analyze Alaska in Detail ---
    print("--- Analyzing Alaska (AK) in Detail ---")
    ak_df = df[df['region'] == 'Alaska']

    if not ak_df.empty:
        # 14a. Alaska Magnitude Distribution
        print("--- Generating Magnitude Distribution for AK ---")
        plt.figure(figsize=(10, 6))
        plt.hist(ak_df['mag'].dropna(), bins=30, edgecolor='black')
        plt.title('Distribution of Earthquake Magnitudes in AK (Last 30 Days)')
        plt.xlabel('Magnitude')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.show()
        print("AK Magnitude plot displayed. Close the plot window to continue.")
        print("\n" + "-"*30 + "\n")

        # 14b. Alaska Magnitude vs. Depth
        print("--- Generating Magnitude vs. Depth for AK ---")
        plt.figure(figsize=(10, 7))
        plt.scatter(ak_df['depth'], ak_df['mag'], alpha=0.5, c=ak_df['mag'], cmap='viridis', s=ak_df['mag']**2)
        plt.colorbar(label='Magnitude')
        plt.title('Earthquake Magnitude vs. Depth in AK (Last 30 Days)')
        plt.xlabel('Depth (km)')
        plt.ylabel('Magnitude')
        plt.grid(True, alpha=0.5)
        plt.show()
        print("AK Magnitude vs. Depth plot displayed. Close the plot window to continue.")
    else:
        print("No data found for AK to analyze.")
    print("\n" + "="*50 + "\n") # Separator

    # --- 15. Time Series of Average Magnitude ---
    print("--- Generating Time Series Plot (Average Magnitude per Day) ---")
    # Resample by day ('D') and calculate mean magnitude
    # Need df_time_indexed from step 5
    if 'df_time_indexed' in locals():
        avg_mag_per_day = df_time_indexed['mag'].resample('D').mean()

        plt.figure(figsize=(12, 6))
        avg_mag_per_day.plot()
        plt.title('Average Earthquake Magnitude per Day (Last 30 Days)')
        plt.xlabel('Date')
        plt.ylabel('Average Magnitude')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        print("Average Magnitude time series plot displayed. Close the plot window to continue.")
    else:
        print("Could not generate average magnitude time series (df_time_indexed not found).")
    print("\n" + "="*50 + "\n") # Separator

    # --- End of Analysis ---
    print("--- End of Analysis ---")

except FileNotFoundError:
    print("Error: The file 'earthquakes_last_30_days.csv' was not found.")
    print("Please ensure the file is in the same directory as the script or provide the correct path.")
except Exception as e:
    print(f"An error occurred: {e}") 