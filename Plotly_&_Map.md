# Plotly Python Cheat Sheet

This cheat sheet provides templates for creating different types of plots using Plotly in Python. Each example includes a function that can be adapted for specific datasets and use cases.

---

## 1. Line Chart
```python
def plot_line_chart(df, x_column, y_column, title):
    """
    Function to create a line chart.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        x_column (str): Column name for x-axis
        y_column (str): Column name for y-axis
        title (str): Title of the chart
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x_column], y=df[y_column], mode='lines', name='Line'))

    fig.update_layout(
        title=title,
        xaxis_title=x_column,
        yaxis_title=y_column,
        template='plotly_white'
    )

    fig.show()
```

---

## 2. Bar Chart
```python
def plot_bar_chart(df, x_column, y_column, title):
    """
    Function to create a bar chart.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        x_column (str): Column name for x-axis
        y_column (str): Column name for y-axis
        title (str): Title of the chart
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Bar(x=df[x_column], y=df[y_column], name='Bar'))

    fig.update_layout(
        title=title,
        xaxis_title=x_column,
        yaxis_title=y_column,
        template='plotly_white'
    )

    fig.show()
```

---

## 3. Scatter Plot
```python
def plot_scatter(df, x_column, y_column, title):
    """
    Function to create a scatter plot.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        x_column (str): Column name for x-axis
        y_column (str): Column name for y-axis
        title (str): Title of the chart
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x_column], y=df[y_column], mode='markers', name='Scatter'))

    fig.update_layout(
        title=title,
        xaxis_title=x_column,
        yaxis_title=y_column,
        template='plotly_white'
    )

    fig.show()
```

---

## 4. Pie Chart
```python
def plot_pie_chart(df, values_column, labels_column, title):
    """
    Function to create a pie chart.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        values_column (str): Column name for pie values
        labels_column (str): Column name for pie labels
        title (str): Title of the chart
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Pie(values=df[values_column], labels=df[labels_column]))

    fig.update_layout(
        title=title,
        template='plotly_white'
    )

    fig.show()
```

---

## 5. Histogram
```python
def plot_histogram(df, x_column, title):
    """
    Function to create a histogram.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        x_column (str): Column name for x-axis
        title (str): Title of the chart
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df[x_column]))

    fig.update_layout(
        title=title,
        xaxis_title=x_column,
        template='plotly_white'
    )

    fig.show()
```

---

## 6. Box Plot
```python
def plot_box_plot(df, y_column, title):
    """
    Function to create a box plot.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        y_column (str): Column name for y-axis
        title (str): Title of the chart
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Box(y=df[y_column], name=y_column))

    fig.update_layout(
        title=title,
        yaxis_title=y_column,
        template='plotly_white'
    )

    fig.show()
```

---

## 7. Subplots
```python
def plot_subplots(df1, df2, x_column1, y_column1, x_column2, y_column2, title):
    """
    Function to create a subplot with two plots.
    
    Args:
        df1, df2 (pd.DataFrame): Input DataFrames
        x_column1, y_column1 (str): Columns for the first plot
        x_column2, y_column2 (str): Columns for the second plot
        title (str): Title of the subplot
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    fig = make_subplots(rows=1, cols=2)

    fig.add_trace(go.Scatter(x=df1[x_column1], y=df1[y_column1], mode='lines', name='Plot 1'), row=1, col=1)
    fig.add_trace(go.Bar(x=df2[x_column2], y=df2[y_column2], name='Plot 2'), row=1, col=2)

    fig.update_layout(
        title=title,
        template='plotly_white'
    )

    fig.show()
```

---

## 8. Choropleth Map
```python
def plot_choropleth_map(df, locations_column, z_column, title, geojson=None):
    """
    Function to create a choropleth map.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        locations_column (str): Column name for locations (ISO country codes, etc.)
        z_column (str): Column name for the color scale
        title (str): Title of the map
        geojson (dict, optional): GeoJSON for custom regions (if needed)
    """
    import plotly.express as px

    fig = px.choropleth(
        df,
        locations=locations_column,
        color=z_column,
        geojson=geojson,
        title=title,
        color_continuous_scale='Viridis'
    )

    fig.update_geos(
        visible=False,
        showcountries=True,
        showcoastlines=True,
        showland=True
    )

    fig.show()
```

---

## 9. Scatter Geo Map
```python
def plot_scatter_geo_map(df, lat_column, lon_column, size_column, hover_name_column, title):
    """
    Function to create a scatter geo map.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        lat_column (str): Column name for latitude
        lon_column (str): Column name for longitude
        size_column (str): Column name for marker size
        hover_name_column (str): Column name for hover information
        title (str): Title of the map
    """
    import plotly.express as px

    fig = px.scatter_geo(
        df,
        lat=lat_column,
        lon=lon_column,
        size=size_column,
        hover_name=hover_name_column,
        title=title,
        projection='natural earth'
    )

    fig.show()
```

---

## 10. Distance and Itinerary Map
```python
def calculate_distance_and_plot_folium(start_coords, end_coords):
    """
    Calculate the road distance between two points using OSM data and plot the itinerary using Folium.

    Args:
        start_coords (tuple): Latitude and longitude of the start point (lat, lon)
        end_coords (tuple): Latitude and longitude of the end point (lat, lon)
    """
    import osmnx as ox
    import networkx as nx
    import folium

    # Increase the search radius to include both points
    G = ox.graph_from_point(start_coords, dist=10000, network_type='drive')

    # Get the nearest nodes to the start and end points
    start_node = ox.distance.nearest_nodes(G, start_coords[1], start_coords[0])
    end_node = ox.distance.nearest_nodes(G, end_coords[1], end_coords[0])

    # Check if nodes exist in the graph
    if start_node not in G or end_node not in G:
        print("One or both points are outside the road network. Increase the search radius.")
        return

    # Calculate the shortest path
    try:
        route = nx.shortest_path(G, start_node, end_node, weight='length')
        route_length = nx.shortest_path_length(G, start_node, end_node, weight='length') / 1000  # Convert meters to km
        print(f"Road Distance: {route_length:.2f} km")
    except nx.NetworkXNoPath:
        print(f"No path exists between the locations: {start_coords} and {end_coords}.")
        return

    # Extract route coordinates
    route_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in route]

    # Calculate the midpoint for displaying distance
    mid_index = len(route_coords) // 2
    midpoint = route_coords[mid_index]

    # Create a Folium map
    itinerary_map = folium.Map(location=start_coords, zoom_start=14)

    # Add start and end points
    folium.Marker(start_coords, popup="Start", icon=folium.Icon(color="green")).add_to(itinerary_map)
    folium.Marker(end_coords, popup="End", icon=folium.Icon(color="red")).add_to(itinerary_map)

    # Add the route
    folium.PolyLine(route_coords, color="blue", weight=2.5).add_to(itinerary_map)

    # Add a popup with the distance at the midpoint
    folium.Marker(
        midpoint,
        popup=f"Distance: {route_length:.2f} km",
        icon=folium.DivIcon(
            html=f'<div style="font-size: 14px; color: blue;">{route_length:.2f} km</div>'
        ),
    ).add_to(itinerary_map)

    # Save and display the Folium map
    itinerary_map.save("itinerary_map.html")
    print("Itinerary map saved as 'itinerary_map.html'")

start_coords = (46.2103456,6.1407283)   # Parking Cornavin
end_coords = (46.2045407,6.1470652)     # Parking Mont-Blanc

calculate_distance_and_plot_folium(start_coords, end_coords)

```

```python
def calculate_distance_and_plot_plotly(start_coords, end_coords):
    """
    Calculate the road distance between two points using OSM data and plot the itinerary using Plotly.

    Args:
        start_coords (tuple): Latitude and longitude of the start point (lat, lon)
        end_coords (tuple): Latitude and longitude of the end point (lat, lon)
    """
    import osmnx as ox
    import networkx as nx
    import plotly.graph_objects as go

    # Increase the search radius to include both points
    G = ox.graph_from_point(start_coords, dist=10000, network_type='drive')

    # Get the nearest nodes to the start and end points
    start_node = ox.distance.nearest_nodes(G, start_coords[1], start_coords[0])
    end_node = ox.distance.nearest_nodes(G, end_coords[1], end_coords[0])

    # Check if nodes exist in the graph
    if start_node not in G or end_node not in G:
        print("One or both points are outside the road network. Increase the search radius.")
        return

    # Calculate the shortest path
    try:
        route = nx.shortest_path(G, start_node, end_node, weight='length')
        route_length = nx.shortest_path_length(G, start_node, end_node, weight='length') / 1000  # Convert meters to km
        print(f"Road Distance: {route_length:.2f} km")
    except nx.NetworkXNoPath:
        print(f"No path exists between the locations: {start_coords} and {end_coords}.")
        return

    # Extract route coordinates
    route_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in route]

    # Create a Plotly map
    lats, lons = zip(*route_coords)
    fig = go.Figure()

    fig.add_trace(go.Scattergeo(
        lat=lats,
        lon=lons,
        mode='lines+markers',
        line=dict(width=2, color='blue'),
        marker=dict(size=10, color=['green'] + ['blue'] * (len(lats) - 2) + ['red']),
        text=['Start'] + ['Route'] * (len(lats) - 2) + ['End']
    ))

    # Update the layout to focus on the itinerary coordinates
    fig.update_layout(
        title=f"Road Itinerary - Distance: {route_length:.2f} km",
        geo=dict(
            scope='europe',
            projection_type='natural earth',
            showland=True,
            center=dict(lat=sum(lats) / len(lats), lon=sum(lons) / len(lons)),  # Center map on the route
            lonaxis=dict(range=[min(lons) - 0.01, max(lons) + 0.01]),  # Zoom longitude
            lataxis=dict(range=[min(lats) - 0.01, max(lats) + 0.01]),  # Zoom latitude
        )
    )

    fig.show()

start_coords = (46.2103456,6.1407283)   # Parking Cornavin
end_coords = (46.2045407,6.1470652)     # Parking Mont-Blanc
calculate_distance_and_plot_plotly(start_coords, end_coords)
```

---

## Notes
- Always ensure your DataFrame is preprocessed before passing it to the functions.
- Customize colors and other properties using the `update_layout` and `update_traces` methods.
- Use `template='plotly_white'` or other templates for consistent styling.
- For Folium maps, ensure you have the `folium` and `geopy` libraries installed.
