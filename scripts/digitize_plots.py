# script to digitize points from figures, likely published spectra.
# written by R. A. Manzuk 04/30/2025
# last updated 12/11/2025

##########################################################################################
# package imports
##########################################################################################
# %%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import os
import numpy as np
# %% 
##########################################################################################
# local function imports
##########################################################################################
# %%
from hyperspectral_u_sensing.utils.misc import get_scale, get_data_points
# %%
###########################################################################################
# script lines
###########################################################################################
# %% header, set up paths, etc.

base_path = '/Users/rmanzuk/Princeton Dropbox/Ryan Manzuk/drc_hyperspectral/curated_spectra/clicked_spectra/figure_pngs'
image_file = 'beiswenger2017_u_minerals.png'     
image_path = os.path.join(base_path, image_file)

# === LOAD IMAGE ===
img = mpimg.imread(image_path)

# %% time to plot and get clicking

fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(img, origin='upper')
ax.set_title("Set axis scales")
plt.grid(True)

# === GET SCALING PARAMETERS ===
x_log = input("Is the X-axis log scale? (y/n): ").lower().strip() == 'y'
x_params = get_scale('x', log_scale=x_log)

y_log = input("Is the Y-axis log scale? (y/n): ").lower().strip() == 'y'
y_params = get_scale('y', log_scale=y_log)

# === GET DATA POINTS AND CONVERT ===
data_points = get_data_points(img, x_params, y_params)

# %% save the data points to a CSV file

# manually change the file name based on what was clicked
out_file_name = 'beiswenger2017_metaautunite.csv' 
output_csv = os.path.join(base_path, out_file_name)
# === SAVE TO CSV ===
df = pd.DataFrame(data_points, columns=['x', 'y'])
df.to_csv(output_csv, index=False)

print(f"\nSaved {len(data_points)} points to '{output_csv}'")
