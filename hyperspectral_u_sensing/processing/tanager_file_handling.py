# set of functions for  handling specifics of tanager data and metadata

# written by R. A. Manzuk 07/28/2025
# last updated 07/30/2025

##########################################################################################
# package imports
##########################################################################################

from affine import Affine
import rasterio.crs
import os

##########################################################################################
# function imports from other files
##########################################################################################

##########################################################################################
# function definitions
##########################################################################################

# ----------------------------------------------------------------------------------------
def find_subpaths(base_dir):
    '''
    Given a base directory, search through the subdirectories for an hdf5 file and a metadata.json file.
    Return the paths to both files.
    If either file is not found, return None for that file. 
    Only works if there is only one hdf5 file and one metadata.json file in the subdirectories.
    If there are multiple files, it will return the last one found.

    Inputs:
    base_dir (str): The base directory to search through.

    Returns:
    hdf5_path (str or None): The path to the hdf5 file, or None if not found.
    metadata_path (str or None): The path to the metadata.json file, or None if not found.

    '''

    hdf5_path = None
    metadata_path = None
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.h5'):
                hdf5_path = os.path.join(root, file)
            elif file.endswith('metadata.json'):
                metadata_path = os.path.join(root, file)
    
    return hdf5_path, metadata_path

# ----------------------------------------------------------------------------------------
def scene_key(ortho=True):
    """
    Create a dictionary with pointers to datasets in a tanager datacube

    Inputs:
    ortho (bool): whether or not the scene is orthorectified

    Returns:
    scene_key (dict): dictionary with pointers to datasets in a tanager datacube    
    """

    # pretty simple, just make a dictionary, depending on ortho or not
    if ortho:
        # basic radiance file key
        scene_key = { 
            'radiance'    : 'HDFEOS/GRIDS/HYP/Data Fields/toa_radiance',
            'lat'         : 'HDFEOS/GRIDS/HYP/Geolocation Fields/Latitude',
            'lon'         : 'HDFEOS/GRIDS/HYP/Geolocation Fields/Longitude',
            'time'        : 'HDFEOS/GRIDS/HYP/Geolocation Fields/Time',
            'cirrus'      : 'HDFEOS/GRIDS/HYP/Data Fields/beta_cirrus_mask',
            'cloud'       : 'HDFEOS/GRIDS/HYP/Data Fields/beta_cloud_mask',
            'nodata'      : 'HDFEOS/GRIDS/HYP/Data Fields/nodata_pixels',
            'pathlen'     : 'HDFEOS/GRIDS/HYP/Data Fields/sensor_to_ground_path_length',
            'vza'         : 'HDFEOS/GRIDS/HYP/Data Fields/sensor_zenith',
            'vaa'         : 'HDFEOS/GRIDS/HYP/Data Fields/sensor_azimuth',
            'sza'         : 'HDFEOS/GRIDS/HYP/Data Fields/sun_zenith',
            'saa'         : 'HDFEOS/GRIDS/HYP/Data Fields/sun_azimuth'
            }
    else:
        scene_key = { 
            'radiance'    : 'HDFEOS/SWATHS/HYP/Data Fields/toa_radiance',
            'lat'         : 'HDFEOS/SWATHS/HYP/Geolocation Fields/Latitude',
            'lon'         : 'HDFEOS/SWATHS/HYP/Geolocation Fields/Longitude',
            'time'        : 'HDFEOS/SWATHS/HYP/Geolocation Fields/Time',
            'cirrus'      : 'HDFEOS/SWATHS/HYP/Data Fields/beta_cirrus_mask',
            'cloud'       : 'HDFEOS/SWATHS/HYP/Data Fields/beta_cloud_mask',
            'nodata'      : 'HDFEOS/SWATHS/HYP/Data Fields/nodata_pixels',
            'pathlen'     : 'HDFEOS/SWATHS/HYP/Data Fields/sensor_to_ground_path_length',
            'vza'         : 'HDFEOS/SWATHS/HYP/Data Fields/sensor_zenith',
            'vaa'         : 'HDFEOS/SWATHS/HYP/Data Fields/sensor_azimuth',
            'sza'         : 'HDFEOS/SWATHS/HYP/Data Fields/sun_zenith',
            'saa'         : 'HDFEOS/SWATHS/HYP/Data Fields/sun_azimuth'
            }

    return scene_key

# ----------------------------------------------------------------------------------------
def load_radiance(scene_file, scene_key):
    """
    Load the radiance data from a tanager scene file.

    Inputs:
    scene_file (h5py.File): the opened HDF5 file containing the tanager scene
    scene_key (dict): dictionary with pointers to datasets in the tanager datacube

    Returns:
    rad_array (numpy.ndarray): array of TOA radiance values
    rad_attributes (dict): attributes of the radiance dataset
    """

    # point to the radiance dataset and load it into memory
    group_id = scene_key['radiance']
    rad_dataset = scene_file[group_id]

    # get the array of hyperspectral values (TOA radiance)
    rad_array = rad_dataset[...]
    rad_attributes = dict(list(rad_dataset.attrs.items()))

    return rad_array, rad_attributes

# ----------------------------------------------------------------------------------------
def h5_metadata(scene_file):
    """
    Get the metadata from a tanager scene file.

    Inputs:
    scene_file (h5py.File): the opened HDF5 file containing the tanager scene

    Returns:
    metadata (dict): dictionary containing the metadata of the scene
    """

    # get the metadata strint and make it legible
    metadata = scene_file['HDFEOS INFORMATION']['StructMetadata.0'][()]
    metadata_str = metadata.decode('utf-8')

    # now it gets a little gnarly, but this is what it will take to parse it
    lines = metadata_str.strip().splitlines()
    stack = []
    root = {}
    current = root

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("GROUP=") or line.startswith("OBJECT="):
            key = line.split("=", 1)[1]
            new_block = {}
            if key in current:
                if isinstance(current[key], list):
                    current[key].append(new_block)
                else:
                    current[key] = [current[key], new_block]
            else:
                current[key] = new_block
            stack.append((current, key))
            current = new_block

        elif line.startswith("END_GROUP=") or line.startswith("END_OBJECT="):
            current, _ = stack.pop()

        else:
            if '=' not in line:
                continue
            k, v = line.split('=', 1)
            k = k.strip()
            v = v.strip()

            # Remove outer quotes
            if v.startswith('"') and v.endswith('"'):
                v = v[1:-1]

            # Handle tuple values
            elif v.startswith('(') and v.endswith(')'):
                contents = v[1:-1].split(',')
                contents = [c.strip().strip('"') for c in contents]
                try:
                    v = tuple(float(c) for c in contents)
                except ValueError:
                    v = tuple(contents)

            else:
                # Try int or float conversion
                if v.isdigit():
                    v = int(v)
                else:
                    try:
                        v = float(v)
                    except ValueError:
                        v = v.strip('"')

            current[k] = v

    return root

# ----------------------------------------------------------------------------------------
def extract_georef_info(parsed_struct, grid_name='HYP'):
    """
    Extract georeferencing information from the parsed metadata structure.

    Inputs:
    parsed_struct (dict): parsed metadata structure from the HDF5 file
    grid_name (str): name of the grid to extract information from (default is 'HYP')

    Returns:
    georef_info (dict): dictionary containing georeferencing information
    """
    grid_structure = parsed_struct.get("GridStructure", {})
    grid = grid_structure.get("GRID_1", {})

    # Sanity check: is this the grid we want?
    if grid.get("GridName") != grid_name:
        raise ValueError(f"Grid name '{grid_name}' not found in GRID_1.")

    required_keys = [
        "XDim", "YDim",
        "UpperLeftPointMtrs", "LowerRightMtrs",
        "Projection", "ZoneCode", "SphereCode"
    ]

    georef_info = {k: grid[k] for k in required_keys if k in grid}

    # Optional extras
    georef_info["PixelRegistration"] = grid.get("PixelRegistration", None)
    georef_info["GridOrigin"] = grid.get("GridOrigin", None)

    return georef_info

# ----------------------------------------------------------------------------------------
def build_affine_transform(georef_info):
    """
    Build an affine transformation from the georeferencing information.
    
    Inputs:
    georef_info (dict): dictionary containing georeferencing information

    Returns:
    affine_transform (Affine): an Affine transformation object
    """

    # Extract necessary values from georef_info
    ulx, uly = georef_info["UpperLeftPointMtrs"]
    lrx, lry = georef_info["LowerRightMtrs"]
    xdim = georef_info["XDim"]
    ydim = georef_info["YDim"]
    
    # Calculate pixel size
    xres = (lrx - ulx) / xdim
    yres = (lry - uly) / ydim  # Note: this will be negative

    # Corner-registered (UL corner of pixel) is default
    pixel_registration = georef_info.get("PixelRegistration", "HE5_HDFE_CORNER")
    if pixel_registration == "HE5_HDFE_CENTER":
        ulx -= xres / 2
        uly -= yres / 2

    return Affine.translation(ulx, uly) * Affine.scale(xres, yres)

# ----------------------------------------------------------------------------------------
def build_crs(georef_info):
    """
    Build a Coordinate Reference System (CRS) object from the georeferencing information.

    Inputs:
    georef_info (dict): dictionary containing georeferencing information

    Returns:
    crs (rasterio.crs.CRS): a CRS object
    """
    # Extract the UTM zone and hemisphere from the georef_info
    zone = abs(georef_info["ZoneCode"])
    south = georef_info["ZoneCode"] < 0

    if not (1 <= zone <= 60):
        raise ValueError(f"Invalid UTM zone number: {zone}")

    epsg = 32700 + zone if south else 32600 + zone
    return rasterio.crs.CRS.from_epsg(epsg)


