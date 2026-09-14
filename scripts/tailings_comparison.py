# script with sections to start looking at and compare spectral characteristics
# of different tailings areas in the DRC hyperspectral data
# written by R. A. Manzuk 11/06/2025
# last updated 11/06/2025
################################################################################
# Analysis outline:
# 1. imports
# 2. load in kolwezi scene and associated polygons
# 3. perform PCA on tailings areas in that scene
# 4. load in broader tailings area polygons
# 5. For each other scene, load it in, mask to tailings areas, and project 
#    onto kolwezi PCA. Store results for comparison.
################################################################################

################################################################################
# Package imports
################################################################################
# %% 

import geopandas as gpd
import h5py
from rasterio.features import rasterize
import matplotlib.pyplot as plt
import numpy as np
from shapely import box
from sklearn.decomposition import PCA
import rasterio as rio
import os

# %%
##########################################################################################
# function imports from other files
##########################################################################################
# %%

import hyperspectral_u_sensing.processing.tanager_file_handling as tanager_file_handling
import hyperspectral_u_sensing.processing.hyperspectral_corrections as hyperspectral_corrections

# %%
##########################################################################################
# script lines
##########################################################################################
# %% define paths and load files

scene_path = '/Users/rmanzuk/Princeton Dropbox/Ryan Manzuk/drc_hyperspectral/tanager_ims_drc/commus-order_tan-ortho-radiance_20250625/files/TanagerScene/20250613_091608_74_4001/radiance_hdf5/20250613_091608_74_4001_ortho_radiance.h5'
scene_file = h5py.File(scene_path, 'r')
scene_key = tanager_file_handling.scene_key(True)
rad_array, rad_attrs = tanager_file_handling.load_radiance(scene_file, scene_key)

polygon_path = '/Users/rmanzuk/Princeton Dropbox/Ryan Manzuk/drc_hyperspectral/traced_polygons/kolwezi_segmentation/kolwezi_polygons.shp'
kolwezi_polygons = gpd.read_file(polygon_path)

# add a line so saved fondts work in pdf plots
plt.rcParams['pdf.fonttype'] = 42

# %% rasterize the polygons at the same resolution as the scene

# this process actually starts by getting geographic metadata from the scene file
metadata = tanager_file_handling.h5_metadata(scene_file)
georef_info = tanager_file_handling.extract_georef_info(metadata)
ref_transform = tanager_file_handling.build_affine_transform(georef_info)
ref_crs = tanager_file_handling.build_crs(georef_info)
ref_shape = (georef_info['YDim'], georef_info['XDim'])

# double check the CRS of the polygons
if kolwezi_polygons.crs != ref_crs:
    kolwezi_polygons = kolwezi_polygons.to_crs(ref_crs)

# make a class mapping dictionary prior to rasterization
unique_classes = kolwezi_polygons['class'].unique()
class_mapping = {cls: i+1 for i, cls in enumerate(sorted(unique_classes))}

# apply the class mapping to the polygons
kolwezi_polygons['class_id'] = kolwezi_polygons['class'].map(class_mapping)

# do the rasterization
shapes = ((geom, value) for geom, value in zip(kolwezi_polygons.geometry, kolwezi_polygons['class_id']))
class_raster = rasterize(
    shapes=shapes,
    out_shape=ref_shape,
    transform=ref_transform,
    fill=0,
    dtype='int32'
)

# %% Let's try masking out bad bands so they aren't distracting for now

# need the mean spectrum to feed into the function
mean_spectrum = np.mean(rad_array, axis=(1,2))

# make the bad band mask and associated bits
bad_mask = hyperspectral_corrections.tanager_bad_band_mask(rad_attrs['wavelengths'],
                          spectrum=mean_spectrum,
                          pad_nm=20)

good_mask = ~bad_mask
bad_indices = np.where(bad_mask)[0].tolist()
rad_array_masked = rad_array[good_mask]
wavelengths_masked = rad_attrs['wavelengths'][good_mask]

# %% and we also need to take care of no data pixels, while we're at it max normalize each band

no_data_val = rad_attrs['_FillValue']
no_data_mask = np.any(rad_array_masked == no_data_val, axis=0)
rad_array_masked[:, no_data_mask] = np.nan

# use the function to max normalize each band
rad_array_masked = hyperspectral_corrections.band_max_norm(rad_array_masked)

# and make a new mean spectrum
mean_spectrum = np.nanmean(rad_array_masked, axis=(1,2))

# %% time to do the PCA on the tailings class

rad_array_reshaped = rad_array_masked.reshape(-1, rad_array_masked.shape[-1] * rad_array_masked.shape[-2]).T

# define the class name to use
class_name = 'tails'
class_ind = class_mapping[class_name]

# make the class raster into a column vector
class_vec = np.ravel(class_raster)

# and create a mask for the class, mask the reshaped rad array
class_mask = class_vec == class_ind
class_array_masked = rad_array_reshaped[class_mask, :]

# %% we can do the pca now

n_components = 5
pca = PCA(n_components=n_components)
pca.fit(class_array_masked)

class_components = pca.components_
class_scores = rad_array_reshaped @ class_components.T

# set any scores outside the class mask to nan
class_scores[~class_mask] = np.nan
class_scores_im = class_scores.T.reshape(n_components, rad_array_masked.shape[1], rad_array_masked.shape[2])

# %% Moving on to generalizing to other scenes

# start by loading in the broader tailings polygons
tailings_polygon_path = '/Users/rmanzuk/Princeton Dropbox/Ryan Manzuk/drc_hyperspectral/traced_polygons/tailings_hypotheses/tailings_hypotheses.shp'
tailings_polygons = gpd.read_file(tailings_polygon_path)

# double check the CRS of the polygons
if tailings_polygons.crs != ref_crs:
    tailings_polygons = tailings_polygons.to_crs(ref_crs)

# %% because we already have an image loaded, we can rasterize the tailings polygons for the kolwezi scene now
# to extract their pc scores in that scene.

# get the scene bounding box
scene_ul = metadata['GridStructure']['GRID_1']['UpperLeftPointMtrs']
scene_lr = metadata['GridStructure']['GRID_1']['LowerRightMtrs']
scene_bounds = box(*scene_ul, *scene_lr)

# figure out which polygons are within the kolwezi scene bounds
tailings_polygons_in = tailings_polygons[tailings_polygons.intersects(scene_bounds)]

# give each of those a unique integer id as a new column
tailings_polygons_in = tailings_polygons_in.reset_index(drop=True)
tailings_polygons_in['poly_id'] = tailings_polygons_in.index + 1

# rasterize those at the same resolution as before, giveting each polygon its poly_id value
shapes = ((geom, value) for geom, value in zip(tailings_polygons_in.geometry, tailings_polygons_in['poly_id']))
tailings_raster = rasterize(
    shapes=shapes,
    out_shape=ref_shape,
    transform=ref_transform,
    fill=0,
    dtype='int32'
)
# %% set up the dict to hold results, and extract the pc scores for each polygon

# make the empty list to hold results, each entry will be a dict for a polygon
results_list = []

# loop through each polygon
for i, row in tailings_polygons_in.iterrows():
    this_dict = {}
    this_dict['owner'] = row['owner']
    this_dict['cobalt_y_n'] = row['cobalt_y_n']
    this_dict['h3po4_y_n'] = row['h3po4_y_n']
    this_dict['notes'] = row['notes']

    # okay, now extract the pc scores for this polygon
    poly_id = row['poly_id']
    poly_mask = tailings_raster.ravel() == poly_id
    
    # might as well extract the spectra from the normalized rad array, and use the components to get pc scores
    poly_spectra = rad_array_reshaped[poly_mask, :]
    poly_pc_scores = poly_spectra @ class_components.T
    this_dict['pc_scores'] = poly_pc_scores
    
    # append this dict to the results list
    results_list.append(this_dict)

# %% Time to get the tailings ponds from the other scenes as well, start by getting all file paths

im_base = '/Users/rmanzuk/Princeton Dropbox/Ryan Manzuk/drc_hyperspectral/tanager_ims_drc'

all_scene_paths = []
all_meta_paths = []

# loop through all the directories in the base
for dirpath, dirnames, filenames in os.walk(im_base):
    for filename in filenames:
        if filename.endswith('_ortho_radiance.h5'):
            all_scene_paths.append(os.path.join(dirpath, filename))
        elif filename.endswith('_metadata.json'):
            all_meta_paths.append(os.path.join(dirpath, filename))

# and delete the kolwezi scene from the list
all_scene_paths = [p for p in all_scene_paths if '20250613_091608_74_4001' not in p]
all_meta_paths = [p for p in all_meta_paths if '20250613_091608_74_4001' not in p]

# double check that they are in the same order by sorting both lists
all_scene_paths = sorted(all_scene_paths)
all_meta_paths = sorted(all_meta_paths)

# %% ready to go through each scene and extract tailings pc scores

for scene_path, meta_path in zip(all_scene_paths, all_meta_paths):
    print(f"Processing scene: {scene_path.split('/')[-1]}")
    
    # load in the scene file
    scene_file = h5py.File(scene_path, 'r')
    scene_key = tanager_file_handling.scene_key(True)
    rad_array, rad_attrs = tanager_file_handling.load_radiance(scene_file, scene_key)
    
    # get the metadata for georef info
    metadata = tanager_file_handling.h5_metadata(scene_file)
    georef_info = tanager_file_handling.extract_georef_info(metadata)
    ref_transform = tanager_file_handling.build_affine_transform(georef_info)
    ref_crs = tanager_file_handling.build_crs(georef_info)
    ref_shape = (georef_info['YDim'], georef_info['XDim'])
    
    # apply the bad band mask
    rad_array_masked = rad_array[good_mask]
    
    # handle no data and max normalize
    no_data_val = rad_attrs['_FillValue']
    no_data_mask = np.any(rad_array_masked == no_data_val, axis=0)
    rad_array_masked[:, no_data_mask] = np.nan
    rad_array_masked = hyperspectral_corrections.band_max_norm(rad_array_masked)
    
    # reshape for pc projection
    rad_array_reshaped = rad_array_masked.reshape(-1, rad_array_masked.shape[-1] * rad_array_masked.shape[-2]).T
    
    # make the tailings raster for this scene
    # figure out which polygons are within the scene bounds
    scene_ul = metadata['GridStructure']['GRID_1']['UpperLeftPointMtrs']
    scene_lr = metadata['GridStructure']['GRID_1']['LowerRightMtrs']
    scene_bounds = box(*scene_ul, *scene_lr)
    tailings_polygons_in = tailings_polygons[tailings_polygons.intersects(scene_bounds)]
    tailings_polygons_in = tailings_polygons_in.reset_index(drop=True)
    tailings_polygons_in['poly_id'] = tailings_polygons_in.index + 1
    shapes = ((geom, value) for geom, value in zip(tailings_polygons_in.geometry, tailings_polygons_in['poly_id']))
    tailings_raster = rasterize(
        shapes=shapes,
        out_shape=ref_shape,
        transform=ref_transform,
        fill=0,
        dtype='int32'
    )   

    # extract the pc scores for each polygon
    for i, row in tailings_polygons_in.iterrows():
        this_dict = {}
        this_dict['scene'] = scene_path.split('/')[-1]
        this_dict['owner'] = row['owner']
        this_dict['cobalt_y_n'] = row['cobalt_y_n']
        this_dict['h3po4_y_n'] = row['h3po4_y_n']
        this_dict['notes'] = row['notes']

        # okay, now extract the pc scores for this polygon
        poly_id = row['poly_id']
        poly_mask = tailings_raster.ravel() == poly_id
        
        # might as well extract the spectra from the normalized rad array, and use the components to get pc scores
        poly_spectra = rad_array_reshaped[poly_mask, :]
        poly_pc_scores = poly_spectra @ class_components.T
        this_dict['pc_scores'] = poly_pc_scores
        
        # append this dict to the results list
        results_list.append(this_dict)

# %% a detailed scatter plot of any 2 PCs for all polygons

pc_x = 2
pc_y = 3

# list out a bunch of symbols to cycle through
symbols = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '8', '<', '>']

# and we'll need counts and colors for y, n, and unknown for h3po4
h3po4_y_count = 0
h3po4_n_count = 0
h3po4_u_count = 0
h3po4_y_color = 'blue'
h3po4_n_color = 'red'
h3po4_u_color = 'gray'

plt.figure(figsize=(8,6))

for res in results_list:
    pc_scores = res['pc_scores']
    h3po4_status = res['h3po4_y_n']
    
    if h3po4_status == 'y':
        symbol = symbols[h3po4_y_count % len(symbols)]
        p_color = h3po4_y_color
        h3po4_y_count += 1
    elif h3po4_status == 'n':
        symbol = symbols[h3po4_n_count % len(symbols)]
        p_color = h3po4_n_color
        h3po4_n_count += 1
    else:
        symbol = symbols[h3po4_u_count % len(symbols)]
        p_color = h3po4_u_color
        h3po4_u_count += 1
    # only give a label if we know the owner
    if res['owner'] != None:
        plt.scatter(pc_scores[:, pc_x], pc_scores[:, pc_y], label=f"{res['owner']} ({h3po4_status})", alpha=0.3, marker=symbol, color=p_color)
    else:
        plt.scatter(pc_scores[:, pc_x], pc_scores[:, pc_y], alpha=0.3, marker=symbol, color=p_color)

plt.xlabel(f'PC {pc_x + 1} Score')
plt.ylabel(f'PC {pc_y + 1} Score')
plt.title('Tailings Polygons Projected onto Kolwezi PCA Space')
plt.grid(True)
plt.legend(title='H3PO4 Status')
plt.tight_layout()
plt.show()

# %% plot the 5 component spectra from the pca

plt.figure(figsize=(10,6))
for i in range(n_components):
    plt.plot(wavelengths_masked, class_components[i], label=f'PC {i+1}')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Component Value')
plt.title('PCA Components for Tailings Class')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
