import cv2
import mediapipe as mp
import os
import pickle
import numpy as np
import time
import sys
import random

# Getting the directory of this file and the model file.
def get_directory() -> str:
    # Getting the directory where this script file is located
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    # Setting the directory where the data is stored, relative to this script file
    DATA_DIR = os.path.join(SCRIPT_DIR, "model.p")
    
    return SCRIPT_DIR, DATA_DIR

# Opening the model file.
def open_model(SCRIPT_DIR, DATA_DIR):
    model_dict = pickle.load(open(DATA_DIR, "rb"))
    model = model_dict['model']
    return model

# Loading the users progress from their practices. 
def load_progress(progress_file:str) -> dict:
    if os.path.exists(progress_file):
        reset_choice = input("You have previously existing progess.\nDo you want to reset your progress?\nEnter 'y' to reset, or 'n' to keep existing progress: ").strip().lower()
        if reset_choice == 'y':
            user_progress = {chr(65+i): {'attempts': 0, 'correct': 0} for i in range(26)}
            save_progress(user_progress, progress_file)
            print("Progress has been reset.")
        elif reset_choice == 'n':
            with open(progress_file, 'rb') as file:
                user_progress = pickle.load(file)
            print("Continuing with existing progress.")
        else:
            print("Invalid input. Continuing with existing progress.")
            with open(progress_file, 'rb') as file:
                user_progress = pickle.load(file)
    else:
        user_progress = {chr(65+i): {'attempts': 0, 'correct': 0} for i in range(26)}

    if input("Would you like to see your progress? (y/n)").lower().strip() == "y":
        print(user_progress)
    return user_progress

# Saving the users progress from their practices. 
def save_progress(progress, file_path) -> None:
    """
    Note: Will have to decide how the progress data will be written and stored.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(progress, file)


# Function to select the next letter to practice
def select_letter(progress):
    """
    Selecting the next letter for practice using a simple heuristic:
    Less accuracy and fewer attempts increase the likelihood of selection.
    """
    letters = list(progress.keys())
    # Calculate weights inversely proportional to accuracy and attempts
    weights = [1 / (progress[letter]['correct'] + 0.1) / (progress[letter]['attempts'] + 1) for letter in letters]
    return random.choices(letters, weights)[0]

def initialize_camera():
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    return cap, hands

def capture_and_process_frame(cap, hands):
    ret, frame = cap.read()
    if not ret:
        sys.exit("Failed to grab frame")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    return frame, results

def make_prediction(model, results, frame):
    data_loc = []  # To store hand landmark data
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                data_loc.extend([x, y])

        prediction = model.predict([np.asarray(data_loc)])
        predicted_character = chr(65 + int(prediction[0]))  
        return predicted_character
    return None

def update_and_display(frame, target_letter, predicted_character):
    is_correct = predicted_character.lower() == target_letter.lower()
    # Display the prediction and target letter on the frame
    cv2.putText(frame, predicted_character, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 3)
    cv2.putText(frame, f"Show this letter: {target_letter}", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Practice Mode", frame)
    return is_correct

def practice_loop(model, progress, file_path):
    cap, hands = initialize_camera()

    # Display initial progress
    print("Initial Progress:")
    print(progress)

    # Added a flag to control the outer loop
    exit_flag = False

    while True:
        target_letter = select_letter(progress)
        print(f"Practice this letter: {target_letter}")
        start_time = time.time()

        while time.time() - start_time < 5:  # 5-second limit for each letter
            frame, results = capture_and_process_frame(cap, hands)
            predicted_character = make_prediction(model, results, frame)
            if predicted_character:
                is_correct = update_and_display(frame, target_letter, predicted_character)
                
                # Update progress after each attempt
                if is_correct:
                    progress[target_letter]['correct'] += 1
                progress[target_letter]['attempts'] += 1
                save_progress(progress, file_path)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit_flag = True
                break

        if exit_flag:
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

    # Display final progress
    print("Final Progress:")
    print(progress)

def main():
    SCRIPT_DIR, DATA_DIR = get_directory()
    model = open_model(SCRIPT_DIR, DATA_DIR)
    progress_file = "user_progress.pkl"
    user_progress = load_progress(progress_file)
    practice_loop(model, user_progress, progress_file)

if __name__ == "__main__":
    main()