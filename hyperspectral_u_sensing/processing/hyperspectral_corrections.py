# set of functions for working with hyperspectral satellite data

# written by R. A. Manzuk 07/28/2025
# last updated 07/29/2025

##########################################################################################
# package imports
##########################################################################################
import numpy as np
from datetime import datetime
import pandas as pd
##########################################################################################
# function imports from other files
##########################################################################################

##########################################################################################
# function definitions
##########################################################################################

# note of Planet github: https://github.com/isofit/isofit

# ----------------------------------------------------------------------------------------
def band_max_norm(img):
    """
    Function to perform maximum normalization on a hyperspectral image. This will normalize
    each band in the image by dividing by the maximum value of that band. This is a common
    normalization technique for hyperspectral data.
    
    Inputs: 
    img (np array): a hyperspectral image with dimensions (bands, rows, cols)
    
    Returns: 
    img_norm (np array): the maximum normalized hyperspectral image
    """
    # get the max of each band
    band_maxes = np.nanmax(img, axis=(1, 2))

    # divide by the max of each band
    img_norm = img / band_maxes[:, np.newaxis, np.newaxis]

    return img_norm

# FUNCTIONS FOR TOA RADIANCE TO REFLECTANCE CONVERSION
# ----------------------------------------------------------------------------------------
def earth_sun_distance(date):
    """
    Function to calculate the Earth-Sun distance in AU for a given date.
    Inputs:
    date (datetime): a datetime object representing the date of interest
    Returns:
    d (float): the Earth-Sun distance in AU
    """

    # approximate formula
    doy = date.timetuple().tm_yday
    return 1.00014 - 0.01671 * np.cos(2 * np.pi * (doy - 3) / 365.25) - 0.00014 * np.sin(2 * np.pi * (doy - 3) / 365.25)

# ----------------------------------------------------------------------------------------
def get_irradiance_vals(wavelengths):
    """
    Function to get the extraterrestrial irradiance values for a given set of wavelengths.
    Using the spreadsheet from ASTM. Hopefully in the future, we will just have direct values
    from planet for Tanager's bands.

    Inputs:
    wavelengths (1d array): an array of wavelengths in nm

    Returns:
    irradiance_vals (1d array): an array of irradiance values in W*m-2*nm-1
    """
    
    # hard coding in the path for now. Again, hopefully this function only is temporary
    irradiance_path = '/Users/rmanzuk/Library/CloudStorage/GoogleDrive-rmanzuk@princeton.edu/My Drive/drc_mining/rem_sensing_data/tanager_ims_drc/irradiance_values_astm_g173_03.csv'

    irradiance_data = pd.read_csv(irradiance_path)
    
    # for each wavelength, get the closest irradiance datapoint
    irradiance_vals = []
    for band in wavelengths:
        closest = (np.abs(irradiance_data['Wavelength (nm)'] - band)).idxmin()
        irradiance_vals.append(irradiance_data['Extraterrestrial W*m-2*nm-1'][closest])
    
    return np.array(irradiance_vals)

# ----------------------------------------------------------------------------------------
def toa_rad_to_reflec(rad_dataset, irradiance_vals, solar_zenith, d):
    """
    Function to convert TOA radiance to TOA reflectance.
    Inputs:
    rad_dataset (np array): a hyperspectral image with dimensions (bands, rows, cols)
    irradiance_vals (1d array): an array of irradiance values in W*m-2*nm-1
    solar_zenith (float): the solar zenith angle in degrees
    d (float): the Earth-Sun distance in AU

    Returns:
    toa_reflectance (np array): the TOA reflectance dataset, samilar to the rad_dataset
    """

    # pretty simple calculation
    cos_theta_s = np.cos(np.radians(solar_zenith))
    toa_reflectance = (np.pi * rad_dataset * d**2) / (irradiance_vals[:, np.newaxis, np.newaxis] * cos_theta_s)
    
    return toa_reflectance

# ----------------------------------------------------------------------------------------
def tanager_dos(toa_reflectance, pctile=1):
    """
    Perform dark object subtraction on a tanager hyperspectral image.

    Inputs:
    toa_reflectance (np array): the TOA reflectance dataset, samilar to the rad_dataset
    pctile (int): the percentile to use for dark object subtraction

    Returns:
    dos_reflectance (np array): the dark object subtracted reflectance dataset
    """
    # flatten for a little speed, and set up output
    flat = toa_reflectance.reshape(toa_reflectance.shape[0], -1)
    dos_reflectance = np.copy(flat)

    # iterating for now, just to easily exclude nans
    for i in range(flat.shape[0]):
        valid_vals = flat[i,~np.isnan(flat[i,:])]
        if len(valid_vals) > 0:
            # get the 1st percentile
            percentile = 1 # hard coded for now
            offset = np.percentile(valid_vals, percentile)
        else:
            offset = 0

    # subtract the offset from the original value
    dos_reflectance[i,:] = np.clip(flat[i] - offset, 0, None)

    # reshape back to original shape
    dos_corrected = dos_reflectance.reshape(toa_reflectance.shape)

    return dos_corrected
# ----------------------------------------------------------------------------------------
def pixelwise_norm(tanager_dataset):
    """
    Function to normalize a tanager hyperspectral image by the geometric mean of each pixel.

    Inputs:
    tanager_dataset (np array): a hyperspectral image with dimensions (bands, rows, cols)

    Returns:
    pixel_normalized (np array): the normalized hyperspectral image
    """
    # flatten for a little speed, and set up output
    flat = tanager_dataset.reshape(tanager_dataset.shape[0], -1)
    normalized_flat = np.copy(flat)

    # because we're using the geometric mean, replace any values of 0 with 1e-6
    flat[flat == 0] = 1e-6

    # compute the geometric mean for each pixel, ignoring nans
    mean_flat = np.exp(np.nanmean(np.log(flat), axis=0))

    # avoid divide by 0
    mean_flat = np.where(mean_flat == 0, np.nan, mean_flat)

    # and normalize
    normalized_flat = flat / mean_flat[np.newaxis, :]

    # clip and reshape
    pixel_normalized = normalized_flat.reshape(tanager_dataset.shape)
    pixel_normalized = np.clip(pixel_normalized, 0, None)
    
    return pixel_normalized
# ----------------------------------------------------------------------------------------
def bandwise_norm(tanager_dataset):
    """
    Function to normalize a tanager hyperspectral image by the geometric mean of each band.

    Inputs:
    tanager_dataset (np array): a hyperspectral image with dimensions (bands, rows, cols)

    Returns:
    band_normalized (np array): the normalized hyperspectral image
    """
    # flatten for a little speed, and set up output
    flat = tanager_dataset.reshape(tanager_dataset.shape[0], -1)
    normalized_flat = np.copy(flat)

    # because we're using the geometric mean, replace any values of 0 with 1e-6
    flat[flat == 0] = 1e-6

    # compute the geometric mean for each band, ignoring nans
    mean_flat = np.exp(np.nanmean(np.log(flat), axis=1))

    # avoid divide by 0
    mean_flat = np.where(mean_flat == 0, np.nan, mean_flat)

    # and normalize
    normalized_flat = flat / mean_flat[:, np.newaxis]

    # clip and reshape
    band_normalized = normalized_flat.reshape(tanager_dataset.shape)
    band_normalized = np.clip(band_normalized, 0, None) 

    return band_normalized
# ----------------------------------------------------------------------------------------
def tanager_bad_band_mask(wavelengths_nm,
                          spectrum=None,
                          pad_nm=10):
    """
    Chat GPT wrote this, so let's see
    Return a boolean mask (True = BAD band to ignore) for Tanager-like hyperspectral data.
    wavelengths_nm: 1D array of band centers in nm (e.g., length ~426)
    spectrum: optional 1D radiance/reflectance array to refine cutoffs
    pad_nm: pad the canonical windows by this many nm on each side
    """
    wl = np.asarray(wavelengths_nm).astype(float)
    bad = np.zeros_like(wl, dtype=bool)

    # --- Canonical "bad" windows (nm), conservative but practical ---
    # Strong Rayleigh/low SNR at the short end
    bad_windows = [
        (350, 420),              # deep UV/blue
        (757, 770),              # O2 A-band
        (930, 965),              # H2O
        (1125, 1165),            # H2O
        (1340, 1465),            # H2O (big)
        (1790, 1990),            # H2O (big)
        (1990, 2070),            # CO2 shoulder (often poor)
        (2380, 2520),            # high-SWIR tail (low SNR)
    ]

    # Apply padding
    for lo, hi in bad_windows:
        lo_p, hi_p = lo - pad_nm, hi + pad_nm
        bad |= (wl >= lo_p) & (wl <= hi_p)

    # --- Optional: data-driven refinement using your spectrum ---
    if spectrum is not None:
        y = np.asarray(spectrum).astype(float)
        if y.shape != wl.shape:
            raise ValueError("spectrum must have same shape as wavelengths_nm")

        # Light smoothing (rolling median) to get a continuum-ish curve
        k = max(5, (len(wl)//100)*2+1)  # odd kernel ~1% of bands, min 5
        pad = k//2
        ypad = np.pad(y, (pad, pad), mode='edge')
        cont = np.array([np.median(ypad[i:i+k]) for i in range(len(y))])

        # Depth relative to local continuum
        depth = (cont - y) / np.maximum(cont, 1e-12)

        # Heuristic: flag bands with strong, sustained depressions
        deep = depth > 0.35      # >35% below continuum
        # merge small gaps so windows are contiguous
        # (dilate once to connect 1-band gaps)
        deep_dilated = deep.copy()
        deep_dilated[1:-1] |= (deep[:-2] & deep[2:])

        # Require at least 3 consecutive bands to avoid speckle
        run = 0
        refine = np.zeros_like(deep_dilated)
        for i, v in enumerate(deep_dilated):
            run = run + 1 if v else 0
            if run >= 3:
                refine[i-run+1:i+1] = True

        # Only add refinement in plausible gas windows to avoid nuking real features
        plausible = (
            ((wl >= 730) & (wl <= 800))  |   # O2 vicinity
            ((wl >= 900) & (wl <= 1000)) |   # H2O ~940
            ((wl >= 1100)& (wl <= 1200)) |   # H2O ~1130
            ((wl >= 1300)& (wl <= 1500)) |   # H2O ~1400
            ((wl >= 1750)& (wl <= 2100)) |   # H2O/CO2 ~1900–2050
            (wl >= 2350)                    # tail
        )
        bad |= (refine & plausible)

    return bad

# Example usage:
# bad_mask = tanager_bad_band_mask(wavelengths_nm, spectrum=None)
# good_mask = ~bad_mask
# bad_indices = np.where(bad_mask)[0].tolist()
