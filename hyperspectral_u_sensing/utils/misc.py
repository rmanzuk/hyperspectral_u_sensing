# set of functions that don't really have a home, but are useful.
# written by R. A. Manzuk 12/11/2025
# last updated 12/11/2025

##########################################################################################
# package imports
##########################################################################################

import matplotlib.pyplot as plt
import numpy as np


###########################################################################################
# other internal function imports
###########################################################################################


###########################################################################################
# function definitions
###########################################################################################

# === USER INPUTS OF PLOT POINTS ===

# Function to get scale and offset for a given axis based on user clicks
def get_scale(axis_name, log_scale=False):
    print(f"Click 2 points along the {axis_name}-axis with known values...")
    plt.title(f"Click 2 points along the {axis_name}-axis")
    clicks = plt.ginput(2, timeout=0)
    px = [p[0] if axis_name == 'x' else p[1] for p in clicks]
    px1, px2 = px

    val1 = float(input(f"Enter the true {axis_name}-axis value at first point: "))
    val2 = float(input(f"Enter the true {axis_name}-axis value at second point: "))

    if log_scale:
        val1, val2 = np.log10([val1, val2])

    scale = (val2 - val1) / (px2 - px1)
    offset = val1 - scale * px1

    return scale, offset, log_scale

# Function to get the data points from the user clicks on the image
def get_data_points(image, x_params, y_params):
    x_scale, x_offset, x_log = x_params
    y_scale, y_offset, y_log = y_params
    print("Click on the data points in the image. Press Enter when done.")

    plt.imshow(image, origin='upper')
    plt.title("Click data points. Press Enter when done.")
    plt.grid(True)
    points = plt.ginput(n=-1, timeout=0)
    plt.close()

    data_points = []
    for x, y in points:
        x_val = x_scale * x + x_offset
        y_val = y_scale * y + y_offset
        if x_log:
            x_val = 10**x_val
        if y_log:
            y_val = 10**y_val
        data_points.append((x_val, y_val))

    return data_points
