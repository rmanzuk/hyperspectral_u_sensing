# general input output stuff

# written by R. A. Manzuk 07/28/2025
# last updated 07/29/2025

##########################################################################################
# package imports
##########################################################################################s
import tkinter as tk
from tkinter import filedialog
import os
##########################################################################################
# function imports from other files
##########################################################################################

##########################################################################################
# function definitions
##########################################################################################

# ----------------------------------------------------------------------------------------
def ui_file_path(start_dir=None):

    # Hide the root window
    root = tk.Tk()
    root.withdraw()

    # sort out the start directory
    initial_dir = start_dir or os.getcwd()

    print("Please select a file using the file picker...")
    file_path = filedialog.askopenfilename(initialdir=initial_dir)

    if file_path:
        print(f"Selected file: {file_path}")
    else:
        print("No file selected.")

    return file_path

# ----------------------------------------------------------------------------------------
def ui_dir_path(start_dir=None):

    # Hide the root window
    root = tk.Tk()
    root.withdraw()

    # sort out the start directory
    initial_dir = start_dir or os.getcwd()

    print("Please select a directory using the file picker...")
    dir_path = filedialog.askdirectory(initialdir=initial_dir)

    if dir_path:
        print(f"Selected file: {dir_path}")
    else:
        print("No file selected.")

    return dir_path



