# script for performing initial explorations of tanager hyperspectral data
# over DRC

# written by R. A. Manzuk 07/28/2025
# last updated 07/28/2025

################################################################################
# Package imports
################################################################################
# %% 
import json
import os
from pprint import pprint
import h5py
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from datetime import datetime
from shapely.geometry import Point
# %%
##########################################################################################
# function imports from other files
##########################################################################################
# %%

from hyperspectral_u_sensing.processing.tanager_file_handling import tanager_scene_key
from hyperspectral_u_sensing.processing.hyperspectral_corrections import get_irradiance_vals, earth_sun_distance, toa_rad_to_reflec, tanager_dos, pixelwise_norm, bandwise_norm
import hyperspectral_u_sensing.utils.tanager_sample_data_utils as tan_utils

# %%
##########################################################################################
# script lines
##########################################################################################
# %% Get directory information for accessing file paths

path_to_ims = '/Users/rmanzuk/Princeton Dropbox/Ryan Manzuk/drc_hyperspectral/tanager_ims_drc'
this_scene = 'shituru-order-tan-ortho-radiance_20250612'
sub_dirs = 'files/TanagerScene/20250606_091508_90_4001'

# %% Read the metadata from the JSON file
metadata_file = os.path.join(path_to_ims, this_scene, sub_dirs, '20250606_091508_90_4001_metadata.json')
with  open(metadata_file, 'r') as f:
    metadata = json.load(f)

# and view it
pprint(metadata)

# %% Set up the path to the data cube, get the scene key

scene_hdf5_path = os.path.join(path_to_ims, this_scene, sub_dirs, 'radiance_hdf5', '20250606_091508_90_4001_ortho_radiance.h5')
scene_file = h5py.File(scene_hdf5_path, 'r')

# create a dictionary with locations of types of data within the cube
# use the function, orthorectification is true
scene_key = tanager_scene_key(True)

# %% using that dictionary, point to the radiance dataset and load it into memory

group_id = scene_key['radiance']
rad_dataset = scene_file[group_id]

# get the array of hyperspectral values (TOA radiance)
rad_array = rad_dataset[...]
rad_attributes=dict(list(rad_dataset.attrs.items()))
print(rad_array.shape)

# %% just show a band at the outset
to_show = 100  # band to show
# no data val is -1e4
rad_array[rad_array < 0] = np.nan  # set no data values to NaN
plt.imshow(rad_array[to_show, :, :], cmap='gray')
plt.colorbar()
plt.title(f'Band {to_show} Radiance')
plt.show()

# %% first correction we need to perform is radiance to reflectance (TOA)

# start by getting irradiance values from ASTM 
irradiance_vals = get_irradiance_vals(rad_attributes['center_wavelengths'])

# get the earth sun distance in AU given the date
date_string = metadata['properties']['acquired']
date = datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%SZ')
d = earth_sun_distance(date)
print(d)

# and solar zenith
solar_zenith = 90 - metadata['properties']['sun_elevation']

# get a toa reflectance dataset
toa_reflectance = toa_rad_to_reflec(rad_array, irradiance_vals, solar_zenith, d)
print(toa_reflectance.shape)

# %% to atmospherically correct, need to start interrogating values double check that things are in order

# just going to set non-data values to NaN
toa_reflectance[toa_reflectance < 0] = np.nan

# %% do DOS

dos_reflectance = tanager_dos(toa_reflectance)

# %% normalize pixelwise

pixel_normalized = pixelwise_norm(dos_reflectance)

# %% normalize each band

band_normalized = bandwise_norm(dos_reflectance)

