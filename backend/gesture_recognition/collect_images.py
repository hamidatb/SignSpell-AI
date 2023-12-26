"""
This code is how I am collecting the images for the data classification.
"""

import os
import cv2

# Get the directory where the script file is located
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

# Set the directory where the data will be stored (inside the backend folder)
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

# Create the directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define the number of classes and the number of images to capture for each class
number_of_classes = 3
dataset_size = 100

# Start capturing video from the webcam (change the index if needed)
cap = cv2.VideoCapture(0)

# Loop over each class
for j in range(number_of_classes):
    # Create a directory for the current class if it doesn't exist
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    # Wait for the user to be ready and press 'Q'
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, 'Press B to begin, Q to quit!', (100, 50), cv2.FONT_HERSHEY_PLAIN, 2, (193, 182, 255), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(25)
        if key == ord('b'):
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break  # Exit the function or use 'break' if in a loop

    # Start capturing images for the current class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
        counter += 1

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
