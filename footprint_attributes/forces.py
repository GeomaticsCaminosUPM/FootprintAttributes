import geopandas as gpd 
import pandas as pd
import shapely 
import numpy as np
import warnings
from utils import get_scaled_normal_vector_at_center, get_angle_90, explode_edges, calculate_momentum

def calc_forces(geoms:gpd.GeoDataFrame,buffer:float=0,height_column:str=None):
    if "force" in geoms.columns:
        warnings.warn("The 'force' column already exists and will be overwritten.", UserWarning)
        
    if "confinement" in geoms.columns:
        warnings.warn("The 'confinement' column already exists and will be overwritten.", UserWarning)
        
    if "momentum" in geoms.columns:
        warnings.warn("The 'momentum' column already exists and will be overwritten.", UserWarning)
        
    if "normalized_angle" in geoms.columns:
        warnings.warn("The 'normalized_angle' column already exists and will be overwritten.", UserWarning)
        
    if "geom_id" in geoms.columns:
        warnings.warn("The 'geom_id' column already exists and will be overwritten.", UserWarning)
        
    orig_crs = geoms.crs
    geoms['geom_id'] = geoms.index.copy()
    geoms_copy = geoms.copy() 
    geoms_copy.geometry = geoms_copy.geometry.force_2d()

    if type(height_column) == type(None):
        geoms_copy['height'] = 1
    else:
        geoms_copy['height'] = geoms_copy[height_column].astype(float)

    if not geoms_copy.crs.is_projected:
        geoms_copy = geoms_copy.to_crs(geoms_copy.geometry.estimate_utm_crs())

    geoms_union = geoms_copy.geometry.union_all().buffer(buffer,cap_style='square',join_style='mitre')
    geoms_union = geoms_union.buffer(-buffer,cap_style='square',join_style='mitre')
    
    geoms_copy.geometry = geoms_copy.geometry.buffer(buffer,cap_style='square',join_style='mitre')
    geoms_copy.geometry = geoms_copy.geometry.buffer(-buffer,cap_style='square',join_style='mitre')
    geoms_copy.geometry = geoms_copy.geometry.boundary.intersection(geoms_union)

    geoms_copy = geoms_copy.loc[geoms_copy.geometry.is_empty == False]

    geoms_copy = explode_edges(geoms_copy,geometry_column='geometry')
    geoms_copy = geoms_copy.loc[geoms_copy['edges'].length > buffer,:]

    geoms_copy[['edge_center','normal']] = geoms_copy.apply(lambda x: pd.Series(get_scaled_normal_vector_at_center(x['edges'],x['height'])),axis=1)

    geoms_copy['momentum'] = geoms_copy.apply(lambda x:calculate_momentum(x['edge_center'],x['normal'],x['centroid']),axis=1)

    geoms_copy['normal_length'] = geoms_copy.apply(lambda x: np.sqrt(x['normal'][0]**2+x['normal'][1]**2),axis=1)
    geoms_copy['normal_length_sqrt'] = geoms_copy['normal_length'] ** 2

    res_normal = geoms_copy.copy()
    res_normal = res_normal.groupby('geom_id').agg({'normal':'sum'}).reset_index()
    res_normal = res_normal.rename(columns={'normal':'res_normal'})
    geoms_copy = geoms_copy.merge(res_normal,on='geom_id',how='left')

    geoms_copy['angle'] = geoms_copy.apply(lambda x: pd.Series(get_angle_90(x['normal'],x['res_normal'],x['geom_id'],x['geom_id'])),axis=1)
    geoms_copy['normalized_angle'] = geoms_copy['angle'] * geoms_copy['normal_length']
    geoms_copy = geoms_copy.groupby('geom_id').agg({
        'height':'first',
        'normal':'sum',
        'normal_length':'sum',
        'angle':'sum',
        'normalized_angle':'sum',
        'normal_length_sqrt':'sum',
        'momentum':'sum',
        'polygon':'first'
    })

    geoms_copy = geoms_copy.rename(columns={'normal':'res_normal'})
    geoms_copy['normal_length_sqrt'] = np.sqrt(geoms_copy['normal_length_sqrt'])
    geoms_copy = geoms_copy.set_geometry('polygon',crs=crs)

    geoms_copy['force'] = geoms_copy.apply(lambda x: np.sqrt(x['res_normal'][0]**2+x['res_normal'][1]**2),axis=1)
    geoms_copy['confinement'] = (geoms_copy['normal_length_sqrt'] - geoms_copy['force'])
    geoms_copy['momentum'] = np.abs(geoms_copy['momentum'])
    geoms_copy['normalized_angle'] = geoms_copy['normalized_angle'] / geoms_copy['normal_length']

    #geoms_copy['area'] = geoms_copy['polygon'].area 
    #geoms_copy['area_sqrt'] = np.sqrt(geoms_copy['polygon'].area)
    #geoms_copy['envelope'] = geoms_copy['polygon'].envelope.length
    geoms_copy = geoms_copy.reset_index()

    result = geoms.merge(geoms_copy[['force','confinement','momentum','normalized_angle']],on='geom_id',how='left')
    result.loc[result['force'].isna(),'force'] = 0 
    result.loc[result['confinement'].isna(),'confinement'] = 0 
    result.loc[result['momentum'].isna(),'momentum'] = 0 
    result.loc[result['normalized_angle'].isna(),'normalized_angle'] = 0 
    result[['force','confinement','momentum','normalized_angle']] = result[['force','confinement','momentum','normalized_angle']].astype(float)
    result.index = result['geom_id'].copy()
    result = result.drop(columns='geom_id')

    return result

def relative_position(footprints: gpd.GeoDataFrame,force_significance: float = 0.05,angle_significance: float = 0.6):
    #footprints['relative_position'] = 'isolated'
    #footprints.loc[(footprints['force'] / footprints['area_sqrt']) > 0.05,'relative_position'] = 'lateral'
    #footprints.loc[(footprints['angle_normalized'] > 0.6) & ((footprints['force'] / footprints['area_sqrt']) > 0.35),'relative_position'] = 'corner'
    #footprints.loc[(footprints['confinement'] / footprints['area_sqrt']) > 0.07,'relative_position'] = 'confined'
    # Warn if relative_position column already exists

    if "relative_position" in footprints.columns:
        warnings.warn(
            "The 'relative_position' column already exists and will be overwritten.", UserWarning
        )
    
    # Preserve original CRS for re-projection if needed
    orig_crs = footprints.crs
    
    # Ensure the geometries are in a projected CRS for accurate calculations
    if not footprints.crs.is_projected:
        footprints = footprints.to_crs(footprints.geometry.estimate_utm_crs())
    
    # Initialize 'relative_position' column
    footprints['relative_position'] = 'isolated'
    
    # Precompute common values for efficiency
    normalized_force = footprints['force'] / np.sqrt(footprints.geometry.area)
    confinement_ratio = footprints['confinement'] / np.sqrt(footprints.geometry.area)
    
    # Update 'relative_position' based on criteria
    footprints.loc[normalized_force > force_significance, 'relative_position'] = 'lateral'
    
    footprints.loc[
        (
            footprints['normalized_angle'] > angle_significance
        ) & (
            normalized_force > force_significance
        ),
        'relative_position'
    ] = 'corner'
    
    footprints.loc[confinement_ratio > force_significance, 'relative_position'] = 'confined'
    
    # Return to the original CRS if needed
    if orig_crs is not None and orig_crs != footprints.crs:
        footprints = footprints.to_crs(orig_crs)
    
    return footprints
