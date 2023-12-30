import os
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import pickle # Used to save datasets and information in binary
from collect_images import get_directory

# Find the landmarks of the hands and return that data
def find_landmarks(SCRIPT_DIR, DATA_DIR):
    # This is the handtracking model from media pipe
    mp_hands = mp.solutions.hands 
    # This is used to draw the hand landmarks on images for visualization.
    mp_drawings = mp.solutions.drawing_utils
    # This offers styling options for the landmarks drawn on the images, such as different colors or line thicknesses.
    mp_drawing_styles = mp.solutions.drawing_styles
    # Setting true means that Im running hand detection on every individual image.
    # Detections below 30% confidence will be ignored.
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    data = []
    labels = []

    # Loop through each subdirectory in the data directory
    for dir_ in os.listdir(DATA_DIR):
        # Construct the full path to the subdirectory
        class_dir = os.path.join(DATA_DIR, dir_)
        # Make sure it's a directory
        if os.path.isdir(class_dir):
            # Loop through each image in the subdirectory
            for img_path in os.listdir(class_dir): # Using slicing to ensure I'm only showing 1
                # Construct the full path to the image
                img_full_path = os.path.join(class_dir, img_path)
                # Make sure it's a file before reading
                if os.path.isfile(img_full_path):
                    # Read the image using OpenCV
                    img = cv2.imread(img_full_path)
                    # Changes the image data into rgb so that matplotlib can actually comprehend whats happening
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    data_loc = []
                    
                    # Iterate over all the landmarks I'v detected in this image
                    results = hands.process(img_rgb) 
                    # Have to make sure we are detecing at least one hand before continuing:
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            for i in range(len(hand_landmarks.landmark)):
                                x = hand_landmarks.landmark[i].x
                                y = hand_landmarks.landmark[i].y

                                data_loc.append(x)
                                data_loc.append(y)
                        
                        data.append(data_loc)
                        labels.append(dir_)

    return data, labels

# Show a visual of the landmarks for the first image of every group
def show_landmarks(SCRIPT_DIR, DATA_DIR):
    # This is the handtracking model from media pipe
    mp_hands = mp.solutions.hands 
    # This is used to draw the hand landmarks on images for visualization.
    mp_drawings = mp.solutions.drawing_utils
    # This offers styling options for the landmarks drawn on the images, such as different colors or line thicknesses.
    mp_drawing_styles = mp.solutions.drawing_styles
    # Setting true means that Im running hand detection on every individual image.
    # Detections below 30% confidence will be ignored.
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

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
                    
                    # Iterate over all the landmarks I'v detected in this image
                    results = hands.process(img_rgb) 
                    # Have to make sure we are detecing at least one hand before continuing:
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            for i in range(len(hand_landmarks.landmark)):
                                x = hand_landmarks.landmark[i].x
                                y = hand_landmarks.landmark[i].y

                        # If you want to see what the drawings on the image look like, you can use the mp.drawings.
                        # For each result we are going to draw the landmarks on top of the image essentially.
                        mp_drawings.draw_landmarks(
                                img_rgb,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS, # The predefined connections between hand landmarks (i.e., how the landmarks are linked to each other).
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style()
                            )

                    plt.figure()
                    plt.imshow(img_rgb) # Loading the image for each image in the path

    # Using MatplotLib to show the images!
    plt.show()
    print("Program ran sucessfully")

# Save the data to a file in binary using the pickle module
def save_file(SCRIPT_DIR, filename, data, labels):
    full_path = os.path.join(SCRIPT_DIR, filename)
    with open (full_path, "wb") as file:
        pickle.dump({"data":data, "labels":labels}, file)
        print(f"File saved successfully to your current folder: {full_path}.")

def main():
    SCRIPT_DIR, DATA_DIR = get_directory()
    data, labels = find_landmarks(SCRIPT_DIR, DATA_DIR)
    save_file(SCRIPT_DIR, "data.pickle", data, labels)
    show_landmarks(SCRIPT_DIR, DATA_DIR) 
    print("This program ran sucessfully.")

if __name__ == "__main__":
    main()