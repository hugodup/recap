# Python Cheat Sheet: Geospatial Data Manipulation and Visualization with Plotly & Folium

## Table of Contents
1. **Installing Necessary Libraries**
2. **Loading Geospatial Data**
3. **Manipulating Geospatial Data**
4. **Analyzing Geospatial Data**
5. **Latitude and Longitude Manipulation**
6. **In-Depth Geospatial Analysis**
7. **Visualizing Geospatial Data with Plotly and Folium**

---

## 1. Installing Necessary Libraries

```bash
pip install geopandas shapely plotly folium
```

## 2. Loading Geospatial Data

### Load Shapefiles and GeoJSON
```python
import geopandas as gpd

# Load shapefile
data = gpd.read_file('path/to/your/shapefile.shp')

# Load GeoJSON
data = gpd.read_file('path/to/your/file.geojson')
```

### Inspect Data
```python
# Print first 5 rows
print(data.head())

# Check CRS (Coordinate Reference System)
print(data.crs)

# Summary of the data
print(data.info())
```

---

## 3. Manipulating Geospatial Data

### Reprojecting CRS
```python
data = data.to_crs('EPSG:4326')  # Reproject to WGS 84 (lat/lon)
```

### Selecting Data by Attributes
```python
# Filter rows where column 'population' > 1000
filtered_data = data[data['population'] > 1000]
```

### Spatial Joins
```python
# Spatial join between two GeoDataFrames
gdf1 = gpd.read_file('file1.geojson')
gdf2 = gpd.read_file('file2.geojson')
joined = gpd.sjoin(gdf1, gdf2, how='inner', op='intersects')
```

### Buffering and Dissolving
```python
# Create a buffer of 100 meters around geometries
data['buffered'] = data.buffer(100)

# Dissolve geometries by a column
dissolved = data.dissolve(by='region')
```

### Exporting Data
```python
# Save to GeoJSON
data.to_file('output.geojson', driver='GeoJSON')

# Save to Shapefile
data.to_file('output.shp')
```

---

## 4. Analyzing Geospatial Data

### Calculate Area and Length
```python
# Area (in CRS units)
data['area'] = data.area

# Length (in CRS units)
data['length'] = data.length
```

### Aggregations
```python
# Group by a column and calculate mean area
grouped = data.groupby('region').agg({'area': 'mean'})
```

### Centroids and Bounds
```python
# Calculate centroids
data['centroid'] = data.centroid

# Calculate bounding boxes
data['bounds'] = data.bounds
```

---

## 5. Latitude and Longitude Manipulation

### Extracting Latitudes and Longitudes from Geometries
```python
# Extract longitude and latitude from point geometries
if data.geometry.geom_type.iloc[0] == 'Point':
    data['longitude'] = data.geometry.x
    data['latitude'] = data.geometry.y
```

### Adding Custom Points Based on Lat/Lon
```python
from shapely.geometry import Point
import pandas as pd

# Create DataFrame with lat/lon
custom_points = pd.DataFrame({
    'name': ['Point A', 'Point B'],
    'latitude': [37.7749, 34.0522],
    'longitude': [-122.4194, -118.2437]
})

# Convert to GeoDataFrame
custom_points['geometry'] = custom_points.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
custom_gdf = gpd.GeoDataFrame(custom_points, geometry='geometry', crs='EPSG:4326')
```

### Calculating Distances Between Points
```python
from shapely.ops import nearest_points

# Distance between two points
point1 = Point(-122.4194, 37.7749)  # San Francisco
point2 = Point(-118.2437, 34.0522)  # Los Angeles

distance = point1.distance(point2)  # In degrees; convert if necessary
print(f"Distance: {distance} degrees")
```

---

## 6. In-Depth Geospatial Analysis

### Nearest Neighbor Analysis
Nearest neighbor analysis is crucial for understanding spatial relationships, such as how close features are to one another. This can help identify clusters, outliers, or trends in spatial data. For example, it can be used in urban planning to determine the proximity of resources to communities or in ecology to study animal movement patterns.

```python
# Find nearest point in one GeoDataFrame to points in another
nearest = custom_gdf.geometry.apply(lambda g: data.distance(g).idxmin())
custom_gdf['nearest_id'] = nearest
custom_gdf['nearest_name'] = custom_gdf['nearest_id'].apply(lambda idx: data.loc[idx, 'name'])
```

### Heatmap Analysis
Heatmaps are a powerful tool for visualizing the density or intensity of features over a geographic area. They are commonly used in fields like crime analysis, retail site selection, and environmental monitoring to detect patterns or hotspots.

```python
import numpy as np
from scipy.stats import gaussian_kde

# Create a heatmap of point density
x = data.geometry.x
y = data.geometry.y
kde = gaussian_kde([x, y])
values = kde([x, y])
data['density'] = values
```

### Spatial Autocorrelation (Moran's I)
Spatial autocorrelation measures the degree to which a spatial variable is correlated with itself through space. Moran's I is commonly used to detect spatial patterns, such as clustering or dispersion. This is valuable in understanding phenomena like disease outbreaks, housing prices, or vegetation cover.

```python
import esda
import libpysal as lps

# Build weights matrix
weights = lps.weights.DistanceBand.from_dataframe(data, threshold=100)

# Calculate Moran's I
moran = esda.moran.Moran(data['population'], weights)
print(f"Moran's I: {moran.I}, p-value: {moran.p_sim}")
```

---

## 7. Visualizing Geospatial Data with Plotly and Folium

### Create a Scatter Plotly Map
```python
import plotly.express as px

# Prepare data
lon = data.geometry.centroid.x
lat = data.geometry.centroid.y

# Add coordinates to the DataFrame
data['lon'] = lon
data['lat'] = lat

# Create Plotly visualization
fig = px.scatter_geo(
    data,
    lat='lat',
    lon='lon',
    color='population',
    size='area',
    hover_name='name',
    title='Geospatial Visualization with Plotly'
)

fig.show()
```

### Choropleth Map with Plotly
```python
# Choropleth map
fig = px.choropleth(
    data,
    geojson=data.geometry,
    locations=data.index,
    color='population',
    title='Population Choropleth Map'
)

fig.show()
```

### Create an Interactive Map with Folium
```python
import folium

# Initialize map
m = folium.Map(location=[37.7749, -122.4194], zoom_start=10)

# Add GeoJSON layer
folium.GeoJson(
    data,
    name='geojson'
).add_to(m)

# Add markers
for _, row in data.iterrows():
    folium.Marker(
        location=[row.geometry.centroid.y, row.geometry.centroid.x],
        popup=f"Name: {row['name']}\nPopulation: {row['population']}"
    ).add_to(m)

# Save map to file
m.save('map.html')

# Display map in a Jupyter Notebook
m
```

---

## Tips

- **CRS Consistency:** Ensure all datasets use the same CRS before spatial operations.
- **Data Cleaning:** Always inspect your data for missing or erroneous values.
- **Performance:** For large datasets, consider using optimized libraries like `Dask` with `Geopandas`.
- **Folium Customization:** Use plugins like `folium.plugins.MarkerCluster` for clustering markers or `folium.LayerControl` for adding map layers.
