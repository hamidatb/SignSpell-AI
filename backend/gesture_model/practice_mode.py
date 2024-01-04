import cv2
import mediapipe as mp
import os
import pickle
import numpy as np
import time
import sys
import random
from hand_gesture_recognizer import recognize_letter
from test_classifier import open_model


# Getting the directory of this file and the model file.
def get_directory() -> str:
    # Getting the directory where this script file is located
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    # Setting the directory where the data is stored, relative to this script file
    DATA_DIR = os.path.join(SCRIPT_DIR, "model.p")
    
    return SCRIPT_DIR, DATA_DIR

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
                x = landmark.x
                y = landmark.y
                data_loc.extend([x, y])

        prediction = model.predict([np.asarray(data_loc)])
        predicted_character = chr(65 + int(prediction[0]))  
        return predicted_character
    return None

def update_and_display(frame, target_letter, predicted_character, amount_remaining):
    is_correct = predicted_character.lower() == target_letter.lower()
    
    # Display the prediction and target letter on the frame
    cv2.putText(frame, f"You are currently showing: {predicted_character}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Show this letter: {target_letter}", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    
    # Display feedback based on whether the prediction is correct
    if is_correct:
        feedback_text = "Correct!"
        feedback_color = (0, 255, 0)  # Green for correct feedback
    else:
        feedback_text = "Incorrect!"
        feedback_color = (0, 0, 255)  # Red for incorrect feedback

    cv2.putText(frame, feedback_text, (50, 150), cv2.FONT_HERSHEY_COMPLEX, 1, feedback_color, 2)
    
    # Display "Amount of letters remaining" on the bottom left
    cv2.putText(frame, f"Amount remaining: {amount_remaining}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display "Press 'q' to quit" on the bottom right
    quit_text = "Press 'q' to quit"
    text_width, _ = cv2.getTextSize(quit_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.putText(frame, quit_text, (frame.shape[1] - text_width - 10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Practice Mode", frame)
    return is_correct

def practice_loop(model, progress, file_path):
    cap, hands = initialize_camera()

    # Adding a flag to control the outer loop (So the user can quit by pressing q)
    exit_flag = False

    while True:
        try:
            amount_of_letters = int(input("How many letters would you like to practice this session?").lower().strip())
            time_wanted = int(input("How much time would you like for each letter?").lower().strip())
            break
        except:
            print("Inavlid amount of letter, or time wanted. Please only submit integer values.")      
    

    for i in range(amount_of_letters):
        target_letter = select_letter(progress)
        print(f"Practice this letter: {target_letter}")
        start_time = time.time()

        amount_remaining = amount_of_letters - 1

        while time.time() - start_time < time_wanted:  # 5-second limit for each letter
            frame, results = capture_and_process_frame(cap, hands)
            predicted_character = make_prediction(model, results, frame)
            if predicted_character:
                is_correct = update_and_display(frame, target_letter, predicted_character, amount_remaining)
                
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

    print(progress)

def main():
    SCRIPT_DIR, DATA_DIR = get_directory()
    model = open_model(SCRIPT_DIR, DATA_DIR)
    progress_file = "user_progress.pkl"
    user_progress = load_progress(progress_file)
    practice_loop(model, user_progress, progress_file)

if __name__ == "__main__":
    main()