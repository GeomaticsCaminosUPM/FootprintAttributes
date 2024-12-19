import geopandas as gpd 
import pandas as pd
import shapely 
import numpy as np
import warnings
from utils import get_scaled_normal_vector_at_center, explode_edges


def calc_shape_irregularity(geoms:gpd.GeoDataFrame):
    if "shape_irregularity" in geoms.columns:
        warnings.warn("The 'shape_irregularity' column already exists and will be overwritten.", UserWarning)
        
    if "geom_id" in geoms.columns:
        warnings.warn("The 'geom_id' column already exists and will be overwritten.", UserWarning)
        
    geoms['geom_id'] = geoms.index.copy()
    geoms_copy = geoms.copy()
    if not geoms_copy.crs.is_projected:
        geoms_copy = geoms_copy.to_crs(geoms_copy.geometry.estimate_utm_crs())

    boundary = geoms_copy.geometry.boundary
    convex_hull = geoms_copy.geometry.convex_hull.boundary
    geoms_copy['irregularity_geom'] = shapely.difference(boundary,convex_hull) #Path through irregularity
    geoms_copy = geoms_copy.loc[geoms_copy['irregularity_geom'].is_empty == False,:]
    geoms_copy['hull'] = shapely.difference(convex_hull,boundary) #Path through hull
    geoms_copy['irregularity_length'] = geoms_copy['irregularity_geom'].length - irregularity['hull'].length
    
    geoms_copy['orig_polygon'] = geoms_copy.geometry.copy()
    geoms_copy = geoms_copy.set_geometry('irregularity_geom',crs=crs).explode(index_parts=False).reset_index(drop=True)

    geoms_copy = explode_edges(geoms_copy,geometry_column='irregularity_geom')
    geoms_copy[['edge_center','normal']] = geoms_copy.apply(lambda x: pd.Series(get_scaled_normal_vector_at_center(x['edges'],1)),axis=1)

    geoms_copy['dist_to_hull'] = geoms_copy.apply(lambda x: shapely.distance(x['edge_center'],x['irregularity_geom']),axis=1)

    geoms_copy.loc[0:len(geoms_copy)-1,'normal_1'] = geoms_copy.loc[1:len(geoms_copy),'normal'].reset_index(drop=True)
    geoms_copy.loc[0:len(geoms_copy)-1,'geom_id_1'] = geoms_copy.loc[1:len(geoms_copy),'geom_id'].reset_index(drop=True)
    #geoms_copy['angle_irregularity'] = geoms_copy.apply(lambda x: pd.Series(get_angle_sharp(x['normal'],x['normal_1'],x['geom_id'],x['geom_id_1'])),axis=1)

    geoms_copy['shape_irregularity'] = geoms_copy['dist_to_hull'] * geoms_copy['edges'].length

    geoms_copy = geoms_copy.groupby('geom_id').agg({
        'shape_irregularity':'sum',
        'irregularity_length':'first',
        'dist_to_hull':'sum',
        #'angle_irregularity':'sum',
        'hull':'first',
        'irregularity_geom':'first'
    })

    #geoms_copy['angle_irregularity'] = geoms_copy['angle_irregularity'] / (np.pi / 2)
    #geoms_copy['angle_irregularity_length'] = geoms_copy['angle_irregularity'] * geoms_copy['irregularity_length']
    geoms_copy = geoms_copy.reset_index()

    result = geoms.merge(geoms_copy[['shape_irregularity','geom_id']],on='geom_id',how='left')
    result.loc[result['shape_irregularity'].isna(),'shape_irregularity'] = 0 
    result['shape_irregularity'] = result['shape_irregularity'].astype(float)
    result.index = result['geom_id'].copy()
    result = result.drop(columns='geom_id')

    return result

def calc_polsby_popper(geoms:gpd.GeoDataFrame):
    # Warn if Polsby-Popper column already exists
    if "polsby_popper" in geoms.columns:
        warnings.warn("The 'polsby_popper' column already exists and will be overwritten.", UserWarning)
        
    # Preserve original CRS for re-projection if needed
    orig_crs = geoms.crs
    
    # Ensure the geometries are in a projected CRS for accurate area and length calculations
    if not geoms.crs.is_projected:
        geoms = geoms.to_crs(geoms.geometry.estimate_utm_crs())
    
    # Calculate the Polsby-Popper compactness score
    geoms['polsby_popper'] = (4 * np.pi * geoms.geometry.area) / (geoms.geometry.boundary.length ** 2)
    
    # Return to the original CRS
    if orig_crs is not None and orig_crs != geoms.crs:
        geoms = geoms.to_crs(orig_crs)
    
    return geoms
    