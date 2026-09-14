# script for taking a tanager scene, and outputting a single band as a GeoTIFF file.
# this script isn't in sections. It's meant to be run as a whole. It will 
# prompt for some inputs.

# written by R. A. Manzuk 07/29/2025
# last updated 07/30/2025

################################################################################
# Package imports (and sys.path appends)
################################################################################

import h5py
import rasterio as rio
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


################################################################################
# User function imports
################################################################################

import hyperspectral_u_sensing.utils.file_io as file_io
import hyperspectral_u_sensing.processing.tanager_file_handling as tanager_file_handling


################################################################################
# Script lines
################################################################################


if __name__ == "__main__":

    # frist print an instruction unique to this script
    print("This script will output a single band from a tanager scene as a GeoTIFF file.")
    print("We'll start by selecting the tanager scene file.")

    # and run the ui file picker function
    selected_path = file_io.ui_file_path()

    # define the scene file
    scene_file = h5py.File(selected_path, 'r')

    # create a dictionary with locations of types of data within the cube
    # use the function, orthorectification is true
    scene_key = tanager_file_handling.scene_key(True)

    # load the radiance dataset and attributes
    rad_array, rad_attrs = tanager_file_handling.load_radiance(scene_file, scene_key)

    # get parsed metadata from the scene file (for geo-referencing)
    metadata = tanager_file_handling.h5_metadata(scene_file)

    # from the metadata, pull out what we need for the GeoTIFF
    georef_info = tanager_file_handling.extract_georef_info(metadata)

    # build the affine transformation matrix, and the CRS
    transform = tanager_file_handling.build_affine_transform(georef_info)
    crs = tanager_file_handling.build_crs(georef_info)

    # before outputting the GeoTIFF, we need some user input and a band number
    band_index = 100 # just being lazy for now
    band_data = rad_array[band_index, :, :]

    # ask the user to input the directory to save the GeoTIFF
    print("Time to get the directory to save the GeoTIFF file.")
    save_directory = file_io.ui_dir_path()

     # and ask the user to input the name of the GeoTIFF file
    geojson_name = input("\nPlease enter the name of the scene: ")
    geojson_name = f"{geojson_name}_{band_index}_band.tif"

    # now save it
    output_path = f"{save_directory}/{geojson_name}"
    with rio.open(
        output_path,
        "w",
        driver="GTiff",
        height=band_data.shape[0],
        width=band_data.shape[1],
        count=1,
        dtype=band_data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(band_data, 1)