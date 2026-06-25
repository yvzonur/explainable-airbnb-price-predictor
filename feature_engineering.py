import pandas as pd
import numpy as np
import json
import math

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    R = 6371.0 # Radius of earth in kilometers
    
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = np.sin(delta_phi / 2.0)**2 + \
        np.cos(phi1) * np.cos(phi2) * \
        np.sin(delta_lambda / 2.0)**2
    
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def compute_polygon_area(coordinates):
    """
    Calculate approximate area of a polygon in sq km using Shoelace formula.
    """
    area = 0.0
    for i in range(len(coordinates) - 1):
        lon1, lat1 = coordinates[i]
        lon2, lat2 = coordinates[i+1]
        
        # Approximate conversion to km
        avg_lat = math.radians((lat1 + lat2) / 2.0)
        x1 = lon1 * 111.320 * math.cos(avg_lat)
        y1 = lat1 * 110.574
        x2 = lon2 * 111.320 * math.cos(avg_lat)
        y2 = lat2 * 110.574
        
        area += (x1 * y2 - x2 * y1)
    return abs(area) / 2.0

def process_geojson(filepath):
    print("Loading GeoJSON...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    areas = {}
    for feature in data['features']:
        props = feature['properties']
        geom = feature['geometry']
        
        # Usually inside properties there is 'neighbourhood' or 'neighbourhood_group'
        # Inside Airbnb provides 'neighbourhood' in the geojson
        name = props.get('neighbourhood')
        if not name:
            continue
            
        if geom['type'] == 'Polygon':
            area = compute_polygon_area(geom['coordinates'][0])
            areas[name] = area
        elif geom['type'] == 'MultiPolygon':
            total_area = 0
            for poly in geom['coordinates']:
                total_area += compute_polygon_area(poly[0])
            areas[name] = total_area
            
    return areas

def run_feature_engineering():
    print("Loading datasets...")
    df = pd.read_csv('listings_cleaned.csv')
    
    # 1. Distance to Center (Taksim Square)
    taksim_lat, taksim_lon = 41.0369, 28.9850
    print("1. Calculating Distance to Center (Taksim)...")
    df['distance_to_center_km'] = haversine(df['latitude'], df['longitude'], taksim_lat, taksim_lon)
    
    # 2. Distance to Bosphorus (Ortakoy point)
    bosphorus_lat, bosphorus_lon = 41.0475, 29.0255
    print("2. Calculating Distance to Bosphorus...")
    df['distance_to_bosphorus_km'] = haversine(df['latitude'], df['longitude'], bosphorus_lat, bosphorus_lon)
    
    # 3. Density calculation
    print("3. Processing GeoJSON for Neighbourhood Density...")
    areas_sq_km = process_geojson('neighbourhoods.geojson')
    
    # In listings_cleaned.csv, neighbourhoods are likely one-hot encoded (e.g. 'Besiktas', 'Kadikoy')
    # Let's count total listings per column and divide by area
    df['listing_density'] = 0.0
    
    columns = set(df.columns)
    for neighbourhood, area in areas_sq_km.items():
        # Check if the neighbourhood has a column in the dataset
        col_name = f'neigh_{neighbourhood}'
        
        if col_name in columns:
            listing_count = df[col_name].sum()
            density = listing_count / area if area > 0 else 0
            
            mask = df[col_name] == 1
            df.loc[mask, 'listing_density'] = density
        elif neighbourhood in columns:
            listing_count = df[neighbourhood].sum()
            density = listing_count / area if area > 0 else 0
            
            mask = df[neighbourhood] == 1
            df.loc[mask, 'listing_density'] = density
            
    print(f"Sample of new features:\n{df[['distance_to_center_km', 'distance_to_bosphorus_km', 'listing_density']].head()}")
    
    output_path = 'listings_with_location.csv'
    df.to_csv(output_path, index=False)
    print(f"\nFeature engineering complete! Saved to {output_path}")

if __name__ == "__main__":
    run_feature_engineering()
