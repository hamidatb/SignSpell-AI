import cv2
import mediapipe as mp
import os
import pickle
import numpy as np
import time
import openai
import sys

# Getting the directory of this file and the model file.
def get_directory():
    # Getting the directory where this script file is located
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    # Setting the directory where the data is stored, relative to this script file
    DATA_DIR = os.path.join(SCRIPT_DIR, "model.p")
    
    return SCRIPT_DIR, DATA_DIR

# Opening the pickle file.
def open_model(SCRIPT_DIR, DATA_DIR):
    model_dict = pickle.load(open(DATA_DIR, "rb"))
    model = model_dict['model']
    return  model

def load_progress(file_path):
    """
    Loads the user's progress from a file.
    :param file_path: The path to the progress file.
    :return: A dictionary containing the user's progress.
    """
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    else:
        return {chr(65+i): {'attempts': 0, 'correct': 0} for i in range(26)}

def save_progress(progress, file_path):
    """
    Saving the user's progress in the practice mode to a file.
    :param progress: The progress dictionary.
    :param file_path: The path to the progress file.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(progress, file)

def practice(model, target_letter):
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    labels_dict = {i: chr(65 + i) for i in range(26)}  # Mapping integers to alphabet letters

    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            sys.exit("Failed to grab frame")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        data_loc = []
        x_ = []
        y_ = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i, landmark in enumerate(hand_landmarks.landmark):
                    x, y = landmark.x, landmark.y
                    data_loc.extend([x, y])
                    x_.append(x)
                    y_.append(y)

            # Generate prediction from the model
            prediction = model.predict([np.asarray(data_loc)])
            predicted_character = labels_dict[int(prediction[0])]

            # Display the prediction on the frame
            cv2.putText(frame, predicted_character, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 3)

        # Display the target letter on the frame
        cv2.putText(frame, f"Show letter: {target_letter}", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("frame", frame)

        elapsed_time = time.time() - start_time
        if elapsed_time > 10:  # 10-second time limit
            break

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Determine if the prediction was correct
    is_correct = predicted_character.lower() == target_letter.lower()
    return is_correct



def play_game(model):
    pass

def main():
    pass

if __name__ == "__main__":
    main()