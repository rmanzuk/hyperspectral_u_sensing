# script with sections to start looking at spectra in a tanager scene, 
# informed by traced polygons
# written by R. A. Manzuk 07/30/2025
# last updated 07/30/2025

################################################################################
# Package imports
################################################################################
# %% 

import geopandas as gpd
import h5py
from rasterio.features import rasterize
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import rasterio as rio

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
polygons = gpd.read_file(polygon_path)

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
if polygons.crs != ref_crs:
    polygons = polygons.to_crs(ref_crs)

# make a class mapping dictionary prior to rasterization
unique_classes = polygons['class'].unique()
class_mapping = {cls: i+1 for i, cls in enumerate(sorted(unique_classes))}

# apply the class mapping to the polygons
polygons['class_id'] = polygons['class'].map(class_mapping)

# do the rasterization
shapes = ((geom, value) for geom, value in zip(polygons.geometry, polygons['class_id']))
class_raster = rasterize(
    shapes=shapes,
    out_shape=ref_shape,
    transform=ref_transform,
    fill=0,
    dtype='int32'
)
# %% also make a raster based on the owner field

# make an owner mapping dictionary prior to rasterization
unique_owners = polygons['owner'].unique()

# if None is in the list, get rid of it
unique_owners = [own for own in unique_owners if own is not None]

owner_mapping = {own: i+1 for i, own in enumerate(sorted(unique_owners))}

# apply the owner mapping to the polygons
polygons['owner_id'] = polygons['owner'].map(owner_mapping)

# do the rasterization
shapes = ((geom, value) for geom, value in zip(polygons.geometry, polygons['owner_id']))
owner_raster = rasterize(
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

# %% make a spaghetti plot of each pixel's spectrum

# it's actually too much to plot all of them, so we'll do a random sample of 5000 pixels
num_pixels = rad_array_masked.shape[1] * rad_array_masked.shape[2]
if num_pixels > 5000:
    sample_indices = np.random.choice(num_pixels, size=5000, replace=False)
    sample_rows = sample_indices // rad_array_masked.shape[2]
    sample_cols = sample_indices % rad_array_masked.shape[2]
    random_sample = rad_array_masked[:, sample_rows, sample_cols]
else:
    random_sample = rad_array_masked

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for i in range(random_sample.shape[1]):
    ax.plot(wavelengths_masked, random_sample[:, i], alpha=0.2)

#ax.plot(wavelengths_masked, mean_spectrum, color='black', linewidth=2, label='Mean Spectrum')

ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Normalized Radiance')
ax.set_title('Spaghetti Plot of Pixel Spectra')
ax.legend()
plt.show()

# %% perform a pca on the masked data

# reshape the array so there is 1 column per band
rad_array_reshaped = rad_array_masked.reshape(-1, rad_array_masked.shape[-1]* rad_array_masked.shape[-2]).T

# remove rows with nans
rad_array_reshaped = rad_array_reshaped[~np.isnan(rad_array_reshaped).any(axis=1)]

n_components = 5
pca = PCA(n_components=n_components)
pca.fit(rad_array_reshaped)

# grab the spectra, we need these to reproject the data.
component_spectra = pca.components_

# %% make a nice plot of just pc2

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(wavelengths_masked, component_spectra[2,:], linewidth=4)
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('PC 3 Loading')
ax.set_title('PCA Component 3 Spectrum')
plt.show()

# %% plot break! Plot the mean spectrum, as well as the spectrum of the five principal components

fix, axs = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
axs = axs.flatten()
axs[0].plot(wavelengths_masked, mean_spectrum, color='black')
axs[1].plot(wavelengths_masked, component_spectra[0,:])
axs[2].plot(wavelengths_masked, component_spectra[1,:])
axs[3].plot(wavelengths_masked, component_spectra[2,:])
axs[4].plot(wavelengths_masked, component_spectra[3,:])
axs[5].plot(wavelengths_masked, component_spectra[4,:])

# put titles on all of them
axs[0].set_title('Mean Spectrum')
for i in range(5):
    axs[i+1].set_title(f'PC {i+1}')

plt.show()

# %% use the components to reproject the masked data

# need to make a new reshaped array
rad_array_reshaped = rad_array_masked.reshape(-1, rad_array_masked.shape[-1]* rad_array_masked.shape[-2]).T

# multiply the reshaped array by the component spectra to get the scores
rad_pc_scores = rad_array_reshaped @ component_spectra.T

# reshape back into an image
rad_pc_image = rad_pc_scores.T.reshape(n_components, rad_array_masked.shape[1], rad_array_masked.shape[2])

# %% show the image of a given component

pc_to_show = 3

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.imshow(rad_pc_image[pc_to_show-1], cmap='gray')

# show a colorbar
fig.colorbar(ax.imshow(rad_pc_image[pc_to_show-1], cmap='gray'), ax=ax, label=f'PC {pc_to_show} Score')

plt.show()

# %% export any component as a GeoTIFF

pc_to_export = 5
pc_data = rad_pc_image[pc_to_export-1]

# spread it to be between 0 and 1
pc_data_min = np.nanmin(pc_data)
pc_data_max = np.nanmax(pc_data)
pc_data = (pc_data - pc_data_min) / (pc_data_max - pc_data_min)

export_path = '/Users/rmanzuk/Princeton Dropbox/Ryan Manzuk/drc_hyperspectral/tanager_ims_drc/pc_geotiffs'
export_name = f'kolwezi_pc{pc_to_export}.tif'
full_export_path = f'{export_path}/{export_name}'

with rio.open(
    full_export_path,
    "w",
    driver="GTiff",
    height=pc_data.shape[0],
    width=pc_data.shape[1],
    count=1,
    dtype=pc_data.dtype,
    crs=ref_crs,
    transform=ref_transform,
) as dst:
    dst.write(pc_data, 1)


# %% now repeat the PCA business, but only within a single class.

rad_array_reshaped = rad_array_masked.reshape(-1, rad_array_masked.shape[-1] * rad_array_masked.shape[-2]).T

# define the class name to use
class_name = 'tank'
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

# %% do the 6 spectra thing

class_mean_spec = np.nanmean(class_array_masked, axis=0)

fig, axs = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
axs = axs.flatten()
axs[0].plot(wavelengths_masked, class_mean_spec, color='black')
axs[1].plot(wavelengths_masked, class_components[0,:])
axs[2].plot(wavelengths_masked, class_components[1,:])
axs[3].plot(wavelengths_masked, class_components[2,:])
axs[4].plot(wavelengths_masked, class_components[3,:])
axs[5].plot(wavelengths_masked, class_components[4,:])

# put titles on all of them
axs[0].set_title('Mean Spectrum')
for i in range(5):
    axs[i+1].set_title(f'PC {i+1}')

plt.show()

# %% plot any component spatially

component_to_show = 2

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.imshow(class_scores_im[component_to_show-1], cmap='gray')
fig.colorbar(ax.imshow(class_scores_im[component_to_show-1], cmap='gray'), ax=ax, label=f'Component {component_to_show} Score')

ax.set_title(f'Component {component_to_show}')

plt.show()

# %% make a cross plot of 2 components, colored by owner

x_comp = 1
y_comp = 4

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for owner_name, owner_id in owner_mapping.items():
    owner_mask = owner_raster == owner_id
    # if this owner has no pixels in the class, skip it
    if not np.any(owner_mask & (class_raster == class_ind)):
        continue
    ax.scatter(class_scores_im[x_comp-1][owner_mask], class_scores_im[y_comp-1][owner_mask], label=owner_name, alpha=0.5, s=30, zorder=1)
ax.set_xlabel(f'Component {x_comp} Score')
ax.set_ylabel(f'Component {y_comp} Score')
ax.set_title(f'Component {x_comp} vs Component {y_comp} Scores Colored by Owner')
# place the legend outside the plot
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout(rect=[0, 0, 0.8, 1])
plt.show()

    
# %% export any component as a GeoTIFF

pc_to_export = 5
pc_data = class_scores_im[pc_to_export-1]

# spread it to be between 0 and 1
pc_data_min = np.nanmin(pc_data)
pc_data_max = np.nanmax(pc_data)
pc_data = (pc_data - pc_data_min) / (pc_data_max - pc_data_min)

export_path = '/Users/rmanzuk/Princeton Dropbox/Ryan Manzuk/drc_hyperspectral/tanager_ims_drc/pc_geotiffs'
export_name = f'kolwezi_{class_name}_pc{pc_to_export}.tif'
full_export_path = f'{export_path}/{export_name}'

with rio.open(
    full_export_path,
    "w",
    driver="GTiff",
    height=pc_data.shape[0],
    width=pc_data.shape[1],
    count=1,
    dtype=pc_data.dtype,
    crs=ref_crs,
    transform=ref_transform,
) as dst:
    dst.write(pc_data, 1)

# %% plot the specrum of any 1 component from the class PCA
component_to_plot = 2
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(wavelengths_masked, class_components[component_to_plot-1,:], linewidth=4)
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel(f'Component {component_to_plot} Loading')
ax.set_title(f'PCA Component {component_to_plot} Spectrum for Class: {class_name}')
plt.show()