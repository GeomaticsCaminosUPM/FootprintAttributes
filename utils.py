import geopandas as gpd 
import pandas as pd
import shapely 
import numpy as np
from shapely.geometry import LineString, Point

# Function to get normal vector at the center of a LineString with scaling factor x
def get_scaled_normal_vector_at_center(linestring, scale=1):
    # 1. Get center point (by length interpolation)
    center_point = linestring.interpolate(0.5, normalized=True)
    
    # 2. Find the two closest points around the center for the tangent
    #midpoint_index = len(linestring.coords) // 2
    p1 = Point(linestring.coords[0])#midpoint_index - 1])
    p2 = Point(linestring.coords[1])#midpoint_index])
    
    # Tangent vector between p1 and p2
    tangent_vector = np.array([p2.x - p1.x, p2.y - p1.y])
    
    # 3. Compute normal vector (perpendicular to tangent)
    normal_vector = np.array([-tangent_vector[1], tangent_vector[0]])

    normal_vector /= np.sqrt(np.sum(normal_vector**2))
    # 4. Scale the normal vector by scale * length of the LineString
    length_of_linestring = linestring.length
    scaled_normal_vector = normal_vector * (scale * length_of_linestring)

    return center_point, scaled_normal_vector

# Function to calculate momentum relative to a reference point
def calculate_momentum(center_point, normal_vector, reference_point):
    r = np.array([center_point.x - reference_point.x, center_point.y - reference_point.y])
    cross_product = r[0] * normal_vector[1] - r[1] * normal_vector[0]
    momentum = cross_product
    
    return momentum

"""
def get_angle_sharp(normal_0,normal_1,geom_id_0,geom_id_1):
    if geom_id_0 != geom_id_1:
        return 0 
     
    _normal_0 = normal_0 / np.sqrt(np.sum(normal_0 ** 2))
    _normal_1 = normal_1 / np.sqrt(np.sum(normal_1 ** 2))
    dot = _normal_0[0] * _normal_1[0] + _normal_0[1] * _normal_1[1]
    cross = _normal_0[0] * _normal_1[1] - _normal_0[1] * _normal_1[0]
    if cross < 0:
        return 0 

    angle = np.arccos(np.abs(dot))
    return angle
"""
def get_angle_90(normal_0,normal_1,geom_id_0=0,geom_id_1=1):
    if geom_id_0 != geom_id_1:
        return 0 
     
    _normal_0 = normal_0 / np.sqrt(np.sum(normal_0 ** 2))
    _normal_1 = normal_1 / np.sqrt(np.sum(normal_1 ** 2))
    dot = _normal_0[0] * _normal_1[0] + _normal_0[1] * _normal_1[1]
    angle = np.arccos(np.abs(dot))

    return angle

def get_angle(normal_0, normal_1, geom_id_0=0, geom_id_1=1):
    # If geometry IDs are different, the angle is undefined (return 0)
    if geom_id_0 != geom_id_1:
        return 0
    
    # Normalize the input vectors
    _normal_0 = normal_0 / np.linalg.norm(normal_0)
    _normal_1 = normal_1 / np.linalg.norm(normal_1)
    
    # Compute the dot and cross products
    dot = np.dot(_normal_0, _normal_1)  # Dot product
    cross = np.cross(_normal_0, _normal_1)  # Cross product (2D equivalent)
    
    # Calculate the angle using arctan2 for accurate quadrant handling
    angle = np.arctan2(cross, dot)
    
    return angle

def explode_edges(geoms,geometry_column:str='geometry'):
    geoms_copy = geoms.copy()
    geoms_copy = geoms_copy.set_geometry(geometry_column,crs=crs)

    coords = geoms_copy.geometry.get_coordinates().reset_index()
    points = gpd.GeoDataFrame(coords['index'],geometry=gpd.points_from_xy(coords['x'],coords['y'])).dissolve('index')
    geoms_copy['split_points'] = points
    geoms_copy['edges'] = geoms_copy.apply(lambda x: shapely.ops.split(x[geometry_column],x['split_points']),axis=1)
    geoms_copy = geoms_copy.set_geometry('edges',crs=crs).explode().reset_index(drop=True)
    geoms_copy = geoms_copy.drop(columns=['split_points'])
    return geoms_copy
