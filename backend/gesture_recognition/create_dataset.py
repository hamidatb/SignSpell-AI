import os
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Getting the directory where this script file is located
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

# Setting the directory where the data is stored, relative to this script file
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# Loop through each subdirectory in the data directory
for dir_ in os.listdir(DATA_DIR):
    # Construct the full path to the subdirectory
    class_dir = os.path.join(DATA_DIR, dir_)
    # Make sure it's a directory
    if os.path.isdir(class_dir):
        # Loop through each image in the subdirectory
        for img_path in os.listdir(class_dir)[:1]: # Using slicing to ensure I'm only showing 1
            # Construct the full path to the image
            img_full_path = os.path.join(class_dir, img_path)
            # Make sure it's a file before reading
            if os.path.isfile(img_full_path):
                # Read the image using OpenCV
                img = cv2.imread(img_full_path)
                # Changes the image data into rgb so that matplotlib can actually comprehend whats happening
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                plt.figure()
                plt.imshow(img_rgb) # Loading the image for each image in the path

# Using MatplotLib to show the images!
plt.show()
 
