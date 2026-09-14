"""
Load all curated spectra CSVs into a single DataFrame, resampled to match
the tanager wavelengths.

Original script by R. A. Manzuk (04/30/2025)
Refactored into a callable module for Jupyter use.
"""

import os
import h5py
import pandas as pd

import hyperspectral_u_sensing.utils.spectrum_data_utils as spectrum_data_utils
import hyperspectral_u_sensing.processing.tanager_file_handling as tanager_file_handling


def load_all_spectra(
    base_path="/Users/rmanzuk/Princeton Dropbox/Ryan Manzuk/drc_hyperspectral/curated_spectra",
    im_path="/Users/rmanzuk/Princeton Dropbox/Ryan Manzuk/drc_hyperspectral/tanager_ims_drc/commus-order_tan-ortho-radiance_20250625"
):
    """
    Loads curated spectra from multiple libraries, resamples them to the Tanager
    wavelength grid, and returns a combined DataFrame containing all spectra.

    Parameters
    ----------
    base_path : str
        Path to the curated_spectra directory.
    im_path : str
        Path to the Tanager imagery directory from which to extract wavelengths.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns: 'wavelengths', <spectrum1>, <spectrum2>, ...
    """

    # ------------------------------------------------------------------
    # Load Tanager wavelengths
    # ------------------------------------------------------------------
    scene_path, _ = tanager_file_handling.find_subpaths(im_path)
    scene_file = h5py.File(scene_path, "r")
    scene_key = tanager_file_handling.scene_key(True)
    _, rad_attrs = tanager_file_handling.load_radiance(scene_file, scene_key)
    tanager_wavelengths = rad_attrs["wavelengths"]

    # Initialize combined DataFrame
    all_spectra_df = pd.DataFrame({"wavelengths": tanager_wavelengths})

    # Folder names
    usgs_subpath = "usgs_splib07"
    klunder_subpath = "klunder2013"
    ecospec_subpath = "ecospec_lib"
    clicked_subpath = "clicked_spectra/spectra"

    # ------------------------------------------------------------------
    # USGS SPLIB07
    # ------------------------------------------------------------------
    usgs_wave_path = os.path.join(base_path, usgs_subpath, "wavelengths")
    usgs_spec_path = os.path.join(base_path, usgs_subpath, "spectra")

    asdfr_wavelengths = spectrum_data_utils.get_usgs_wavelengths(usgs_wave_path, "ASDFR")
    aviris_wavelengths = spectrum_data_utils.get_usgs_wavelengths(usgs_wave_path, "AVIRIS")
    beckman_wavelengths = spectrum_data_utils.get_usgs_wavelengths(usgs_wave_path, "BECK")
    nicolet_wavelengths = spectrum_data_utils.get_usgs_wavelengths(usgs_wave_path, "NIC4")

    usgs_spectra_files = os.listdir(usgs_spec_path)

    for spec_file in usgs_spectra_files:
        spec_path = os.path.join(usgs_spec_path, spec_file)

        with open(spec_path, "r") as f:
            spec_data = f.readlines()

        # metadata and sample name
        spec_metadata = spec_data[0].split()
        spec_name = spec_metadata[2]

        # spectrum values
        this_spectrum = [float(line.split()[0]) for line in spec_data[1:]]

        # choose wavelength grid
        if len(this_spectrum) == len(asdfr_wavelengths):
            these_x = asdfr_wavelengths
        elif len(this_spectrum) == len(aviris_wavelengths):
            these_x = aviris_wavelengths
        elif len(this_spectrum) == len(beckman_wavelengths):
            these_x = beckman_wavelengths
        elif len(this_spectrum) == len(nicolet_wavelengths):
            these_x = nicolet_wavelengths
        else:
            continue

        # resample and add
        resampled = spectrum_data_utils.resample_spectrum(these_x, this_spectrum, tanager_wavelengths)
        all_spectra_df[spec_name] = resampled

    # ------------------------------------------------------------------
    # Klunder 2013
    # ------------------------------------------------------------------
    klunder_path = os.path.join(base_path, klunder_subpath)
    klunder_csv = [f for f in os.listdir(klunder_path) if f.endswith(".csv")][0]
    klunder_csv_path = os.path.join(klunder_path, klunder_csv)

    klunder_data = pd.read_csv(klunder_csv_path, header=None)
    klunder_wavelengths = klunder_data.iloc[0, 1:].values.astype(float)

    for idx in range(1, klunder_data.shape[0]):
        sample_name = klunder_data.iloc[idx, 0]
        sample_spectrum = klunder_data.iloc[idx, 1:].values.astype(float)

        resampled = spectrum_data_utils.resample_spectrum(
            klunder_wavelengths, sample_spectrum, tanager_wavelengths
        )
        all_spectra_df[sample_name] = resampled

    # ------------------------------------------------------------------
    # Ecospec library
    # ------------------------------------------------------------------
    ecospec_path = os.path.join(base_path, ecospec_subpath)
    ecospec_files = [f for f in os.listdir(ecospec_path) if f.endswith("spectrum.txt")]

    for ecospec_file in ecospec_files:
        ecospec_file_path = os.path.join(ecospec_path, ecospec_file)

        ecospec_dict = spectrum_data_utils.read_ecospec_txt(ecospec_file_path)
        sample_name = ecospec_dict["Name"]
        ecospec_wavelengths = ecospec_dict["spectrum"][0] * 1000  # microns → nm
        ecospec_spec = ecospec_dict["spectrum"][1]

        resampled = spectrum_data_utils.resample_spectrum(
            ecospec_wavelengths, ecospec_spec, tanager_wavelengths
        )
        all_spectra_df[sample_name] = resampled

    # ------------------------------------------------------------------
    # Clicked spectra
    # ------------------------------------------------------------------
    clicked_path = os.path.join(base_path, clicked_subpath)
    clicked_files = [f for f in os.listdir(clicked_path) if f.endswith(".csv")]

    for clicked_file in clicked_files:
        clicked_file_path = os.path.join(clicked_path, clicked_file)
        clicked_data = pd.read_csv(clicked_file_path)

        clicked_wavelengths = clicked_data.iloc[:, 0].values.astype(float)
        clicked_spectrum = clicked_data.iloc[:, 1].values.astype(float)
        sample_name = os.path.splitext(clicked_file)[0]

        resampled = spectrum_data_utils.resample_spectrum(
            clicked_wavelengths, clicked_spectrum, tanager_wavelengths
        )
        all_spectra_df[sample_name] = resampled

    return all_spectra_df


if __name__ == "__main__":
    df = load_all_spectra()
    print(df.head())
