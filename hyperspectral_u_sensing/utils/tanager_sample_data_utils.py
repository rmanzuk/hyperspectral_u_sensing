# Code developed by the Planet Tanager team in support of our sample data cubes.
# This code is provided as is, with no guarantees it will work for your use case.

# Import libraries
import h5py
import re
import numpy as np

# Define functions
def print_contents(name, node):
    """
    Print the names of the locations in the h5 file where data is stored.

    Args:
        name (str): The name of the location in the file.
        node (h5py.Group): The group object in the file.

    Returns:
        None
    """
    if isinstance(node, h5py.Dataset):
        print(name)
    return


def print_nodes(name, node):
    """
    Print the name, shape, and data type of each dataset in the h5 file.

    Args:
        name (str): The name of the dataset.
        node (h5py.Dataset): The dataset object.

    Returns:
        None
    """
    if isinstance(node, h5py.Dataset):
        print(node)
    return


def ls_dataset(name, node):
    """
    Print the names of the datasets and any attributes attached to each one.

    Args:
        name (str): The name of the dataset. (optional)
        node (h5py.Dataset): The dataset object.

    Returns:
        None
    """
    if isinstance(node, h5py.Dataset):
        print(node)
        # see if there are attributes attached to the dataset
        attrs = {key: val for key, val in node.attrs.items()}
        print(attrs)
        print()
    return


def find_closest_band(wavelengths, target_wavelength):
    """
    Find the band closest to a target wavelength.

    Args:
        wavelengths (array): Array of wavelengths.
        target_wavelength (float): Target wavelength.

    Returns:
        int: Index of the closest band
    """
    return (np.abs(wavelengths - target_wavelength)).argmin()

def band_normalize(band_data):
    """
    Normalize band data to the range [0, 1] using the minimum and maximum values.

    Args:
        band_data (array): The band data to normalize.

    Returns:
        norm_data (array): The normalized band data
    """
    data_min = np.min(band_data)
    data_max = np.max(band_data)
    norm_data = (band_data - data_min) / (data_max - data_min)
    return norm_data