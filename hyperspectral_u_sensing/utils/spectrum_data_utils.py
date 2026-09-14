# set of functions for dealing with spectral libraries, from data reading to analysis, as
# well as resampling spectra to match satellite sensor wavelengths
##########################################################################################

# written by R. A. Manzuk 01/29/2025
# last updated 12/11/2025

##########################################################################################
# package imports
##########################################################################################
import numpy as np # for array operations
import os # for file path operations
##########################################################################################
# function imports from other files
##########################################################################################

##########################################################################################
# function definitions
##########################################################################################

# ----------------------------------------------------------------------------------------

def extract_ecospec_field(ecospec_data, field_name):
    '''
    Function to extract a single field from an ecospec data text file

    Inputs:
    ecospec_data: the list of strings from reading in the ecospec data
    field_name: a string of the field to extract, case sensitive

    Returns:
    field_data: a list of the data from the field
    '''
    field_line = [line for line in ecospec_data if field_name in line]

    # one check, if there was no data for that field, return None
    if len(field_line) == 0:
        return None
    else:
        field_line = field_line[0]
        field_data = field_line.split(':')[1].strip()
    return field_data

# ----------------------------------------------------------------------------------------
def extract_ecospec_spectrum(ecospec_data):
    '''
    Function to extract the spectral data from an ecospec data text file

    Inputs:
    ecospec_data: the list of strings from reading in the ecospec data

    Returns:
    spectrum_x: a list of the x values of the spectrum
    spectrum_y: a list of the y values of the spectrum
    '''
    # the spectrum will be all the lines after a line that is just ' \n' or '\n' or '\t\n'
    start_line = [i for i, line in enumerate(ecospec_data) if line == ' \n' or line == '\n' or line == '\t\n'][0]
    init_spectrum = ecospec_data[start_line + 1:]

    # now split and strip the data, put into x and y lists
    spectrum_data = [line.strip().split() for line in init_spectrum]
    spectrum_x = [float(pair[0]) for pair in spectrum_data]
    spectrum_y = [float(pair[1]) for pair in spectrum_data]

    return spectrum_x, spectrum_y

# ----------------------------------------------------------------------------------------
def read_ecospec_txt(ecospec_path, fields = ['Name', 'Class', 'Subclass', 'Sample No.',
                                              'Description', 'X Units', 'Y Units']):
    '''
    Function to read in an ecospec text file and return the data
    formatted into a dictionary with the desired fields. The function
    asssumes the user wants the spectum, so that automatically gets
    added to the dictionary

    Inputs:
    ecospec_path: a string of the path to the ecospec text file

    Returns:
    ecospec_data: a list of strings from reading in the file
    '''
    # read the text file
    with open(ecospec_path, 'r', errors='replace') as file:
        ecospec_data = file.readlines()
    
    # put in the desired metadata fields
    ecospec_dict = {}
    for field in fields:
        ecospec_dict[field] = extract_ecospec_field(ecospec_data, field)

    # add the spectrum
    spectrum_x, spectrum_y = extract_ecospec_spectrum(ecospec_data)
    ecospec_dict['spectrum'] = np.array([spectrum_x, spectrum_y])

    return ecospec_dict

# ----------------------------------------------------------------------------------------
def get_usgs_wavelengths(usgs_wavelength_path, ref):
    '''
    Function to go to the right spot in the USGS spectral library, read in
    the text file for the desired set of reference wavelenths, and return
    the wavelengths as a list. NOTE: this function assumes we are using the
    splib07b library (oversampled)

    Inputs:
    usgs_path: a string of the path to the USGS spectral library ASCII data 
    of choice
    ref: a string of the reference wavelength set to extract. options are:
        - ASDFR
        - AVIRIS
        - BECK
        - NIC4

    Returns:
    wavelengths: a list of the wavelengths
    '''
    # double check ref is in the options
    if ref not in ['ASDFR', 'AVIRIS', 'BECK', 'NIC4']:
        raise ValueError('ref must be one of: ASDFR, AVIRIS, BECK, NIC4')
    
    # read the text file
    if ref == 'ASDFR':
        ref_file = os.path.join(usgs_wavelength_path, 'splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt')
    elif ref == 'AVIRIS':
        ref_file = os.path.join(usgs_wavelength_path, 'splib07a_Wavelengths_AVIRIS_1996_0.37-2.5_microns.txt')
    elif ref == 'BECK':
        ref_file = os.path.join(usgs_wavelength_path, 'splib07a_Wavelengths_BECK_Beckman_0.2-3.0_microns.txt')
    elif ref == 'NIC4':
        ref_file = os.path.join(usgs_wavelength_path, 'splib07a_Wavelengths_NIC4_Nicolet_1.12-216microns.txt')

    with open(ref_file, 'r') as file:
        raw_data = file.readlines()

    # with these we can cheat and know they are on the 2nd line and beyond
    wavelengths = [float(line.split()[0]) for line in raw_data[1:]]
    return wavelengths

# ----------------------------------------------------------------------------------------
def read_all_usgs(usgs_path):
    '''
    Function to read in all the USGS spectral library data from a directory
    and return it as a list of dictionaries. Could be nice to have a function
    to just read a single text file, but with the way the data is structured,
    this function is what I need for now.

    Inputs:
    usgs_path: a string of the path to the directory containing the USGS spectral
    library data

    Returns:
    usgs_dicts: a list of dictionaries, each containing the metadata and spectrum
    data for all usgs samples in the directory
    '''
    # start by getting the possible reference wavelengths
    asdfr_wavelengths = get_usgs_wavelengths(usgs_path, 'ASDFR')
    aviris_wavelengths = get_usgs_wavelengths(usgs_path, 'AVIRIS')
    beckman_wavelengths = get_usgs_wavelengths(usgs_path, 'BECK')
    nicolet_wavelengths = get_usgs_wavelengths(usgs_path, 'NIC4')

    # now ready to roll onto spectra, they are in subdirectories of the main directory, that all star with 'Chapter'
    usgs_subdirs = [os.path.join(usgs_path, subdir) for subdir in os.listdir(os.path.join(usgs_path)) if subdir.startswith('Chapter')]

    # we'll loop through the subdirectories, and then through the files in each subdirectory
    usgs_dicts = []
    for subdir in usgs_subdirs:
        usgs_files = [os.path.join(subdir, file) for file in os.listdir(subdir) if file.endswith('txt')]
    
        # store the chapter name so we can use it as the class, it's after the last underscore
        chapter_name = subdir.split('_')[-1]
        for file in usgs_files:
        
            this_dict = {}

            # now we can read in the data
            with open(file, 'r') as file:
                usgs_data = file.readlines()

            # we can fill in the metadata from the first line, split that by the spaces
            metadata = usgs_data[0].split()
            # the 3rd entry is the name
            this_dict['Name'] = metadata[2]
            # the chapter is the class
            this_dict['Class'] = chapter_name
            # we won't have subclass
            this_dict['Subclass'] = None
            # the sample number is the 2nd entry, but after the =
            this_dict['Sample No.'] = metadata[1].split('=')[-1]
            # no description
            this_dict['Description'] = None
            # the x units are 'Wavelength (micrometers)'
            this_dict['X Units'] = 'Wavelength (micrometers)'
            # the y units are 'Reflectance'
            this_dict['Y Units'] = 'Reflectance'
            # and then we can collage the spectrum, with the wavelengths as the x values and the reflectance as the y values
            this_spectrum = [float(line.split()[0]) for line in usgs_data[1:]]

            # based upon the length of the spectrum, we can determine which wavelength data to use
            if len(this_spectrum) == len(asdfr_wavelengths):
                these_x = asdfr_wavelengths
            elif len(this_spectrum) == len(aviris_wavelengths):
                these_x = aviris_wavelengths
            elif len(this_spectrum) == len(beckman_wavelengths):
                these_x = beckman_wavelengths
            elif len(this_spectrum) == len(nicolet_wavelengths):
                these_x = nicolet_wavelengths

            this_dict['spectrum'] = np.array([these_x, this_spectrum])

            usgs_dicts.append(this_dict)

    return usgs_dicts

# ----------------------------------------------------------------------------------------
# Function: resample_spectrum
# Author: ChatGPT (OpenAI)
# Date: 2025-12-11
def resample_spectrum(orig_wavelengths, orig_reflectance, new_wavelengths, fill_value=np.nan):
    """
    Linearly resample a reflectance spectrum to a new wavelength grid
    without extrapolating beyond the original wavelength range.

    Parameters
    ----------
    orig_wavelengths : array-like
        1D array of wavelengths for the original spectrum (must be monotonic).
    orig_reflectance : array-like
        1D array of reflectance values corresponding to `orig_wavelengths`.
    new_wavelengths : array-like
        1D array of target wavelengths to resample onto.
    fill_value : float, optional
        Value to assign to wavelengths outside the original range. Default is np.nan.

    Returns
    -------
    resampled : np.ndarray
        1D array of reflectance values on the new wavelength grid.
    """

    orig_wavelengths = np.asarray(orig_wavelengths)
    orig_reflectance = np.asarray(orig_reflectance)
    new_wavelengths = np.asarray(new_wavelengths)

    # Perform interpolation
    interpolated = np.interp(
        new_wavelengths,
        orig_wavelengths,
        orig_reflectance,
    )

    # Mask out-of-bounds values
    out_of_bounds = (new_wavelengths < orig_wavelengths.min()) | \
                    (new_wavelengths > orig_wavelengths.max())

    interpolated[out_of_bounds] = fill_value

    return interpolated
