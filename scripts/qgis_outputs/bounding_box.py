# script for taking a tanager scene, getting the bounding box, and saving it 
# to a geojson file. 
# this script isn't in sections. It's meant to be run as a whole. It will 
# prompt for some inputs.

# written by R. A. Manzuk 07/29/2025
# last updated 07/29/2025

################################################################################
# Package imports (and sys.path appends)
################################################################################

import json
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

################################################################################
# User function imports
################################################################################

import hyperspectral_u_sensing.utils.file_io as file_io


################################################################################
# Script lines
################################################################################


if __name__ == "__main__":

    # frist print an instruction unique to this script
    print("This script will create a bounding box from a tanager scene and save it to a shapefile.")
    print("We'll start by selecting the metadata.json file for the scene.")

    # and run the ui file picker function
    selected_path = file_io.ui_file_path()

    # read in the metadata file
    with open(selected_path, 'r') as f:
        metadata = json.load(f)

    # and then the bounding box is in 'geometry'
    bounding_geojson = metadata['geometry']

    # time to prep for saving the geojson
    # get just the file name from the path
    in_file_name = selected_path.split('/')[-1]

    # in the file name for the geojson, grab everything before _metadata.json
    file_numbers = in_file_name.split('_metadata.json')[0]

    # ask the user to input the directory to save the geojson
    print("Time to get the directory to save the geojson file.")
    save_directory = file_io.ui_dir_path()

    # and ask the user to input the name of the geojson file
    geojson_name = input("\nPlease enter the name of the scene: ")
    geojson_name = f"{geojson_name}_{file_numbers}_bounding_box.geojson"

    # now save the geojson file
    geojson_path = f"{save_directory}/{geojson_name}"
    with open(geojson_path, 'w') as f:
        json.dump(bounding_geojson, f)
