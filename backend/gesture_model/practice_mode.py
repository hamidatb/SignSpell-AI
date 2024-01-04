import cv2
import mediapipe as mp
import os
import numpy as np
import time
import sys
import random
from hand_gesture_recognizer import recognize_letter
from test_classifier import open_model
from mode_settings import load_progress, save_progress, practice_settings


# Getting the directory of this file and the model file.
def get_directory() -> str:
    # Getting the directory where this script file is located
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    # Setting the directory where the data is stored, relative to this script file
    DATA_DIR = os.path.join(SCRIPT_DIR, "model.p")
    
    return SCRIPT_DIR, DATA_DIR

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

def update_and_display(frame, target_letter, predicted_character, amount_remaining, time_remaining):
    white = (255, 255, 255)
    green = (0, 255, 0)
    red = (0, 0, 255)
    black = (0, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_bottom_margin = 30  # Margin from the bottom of the frame

    # Clear previous text by drawing a filled rectangle
    cv2.rectangle(frame, (0, frame.shape[0] - 100), (frame.shape[1], frame.shape[0]), black, -1)

    # Check if a prediction was made
    if predicted_character is not None:
        is_correct = predicted_character.lower() == target_letter.lower()
        feedback_text = "Correct!" if is_correct else "Incorrect!"
        feedback_color = green if is_correct else red
        cv2.putText(frame, f"Currently showing: {predicted_character}", (10, 30), font, 0.7, white, 2)
    else:
        is_correct = False
        feedback_text = "No prediction"
        feedback_color = red
        cv2.putText(frame, "Currently showing: N/A", (10, 30), font, 0.7, white, 2)

    # Display the target letter, feedback, and remaining amount at the top
    cv2.putText(frame, f"Show this letter: {target_letter}", (10, 70), font, 0.7, white, 2)
    cv2.putText(frame, feedback_text, (300, 70), font, 0.7, feedback_color, 2)
    
    # Display remaining amount and time at the bottom
    cv2.putText(frame, f"Amount remaining: {amount_remaining}", (10, frame.shape[0] - text_bottom_margin), font, 0.7, white, 2)
    cv2.putText(frame, f"Seconds remaining: {time_remaining:.2f}s", (300, frame.shape[0] - text_bottom_margin), font, 0.7, white, 2)

    # Display quit instruction on the bottom right
    quit_text = "Press 'q' to quit"
    text_width, _ = cv2.getTextSize(quit_text, font, 0.7, 2)[0]
    cv2.putText(frame, quit_text, (frame.shape[1] - text_width - 10, frame.shape[0] - text_bottom_margin), font, 0.7, white, 2)
    
    cv2.imshow("Practice Mode", frame)
    return is_correct


def practice_loop(model, progress, file_path, settings):
    cap, hands = initialize_camera()

    # Adding a flag to control the outer loop (So the user can quit by pressing q)
    exit_flag = False

    amount_of_letters = settings["Amount of letters to practice"]
    time_wanted = settings["Time for each letter (seconds)"]

    amount_remaining = amount_of_letters - 1

    marks = {}
    
    for i in range(amount_of_letters):
        target_letter = select_letter(progress)
        print(f"Practice this letter: {target_letter}")
        start_time = time.time()

        while True:
            time_elapsed = time.time() - start_time
            time_remaining = time_wanted - time_elapsed
            if time_remaining <= 0:
                break  # Exit the loop if the time is up

            frame, results = capture_and_process_frame(cap, hands)
            predicted_character = make_prediction(model, results, frame)
            update_and_display(frame, target_letter, predicted_character, amount_remaining, time_remaining)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                exit_flag = True
                break
            else:
                if predicted_character:
                    end_time = time.time()
                    time_taken = round(end_time - start_time, 2)
                    is_correct = (predicted_character.lower() == target_letter.lower())
                    if is_correct:
                        progress[target_letter]['correct'] += 1
                        progress[target_letter]['times'].append(time_taken)
                        marks[target_letter] = ("Correct", time_taken)
                        break  # Break out of the while loop once correct prediction is made
                    else:
                        marks[target_letter] = ("Incorrect", time_taken)
                else:
                    if target_letter not in marks or marks[target_letter][0] != "Correct":
                        marks[target_letter] = ("Incorrect", time_wanted)


            progress[target_letter]['attempts'] += 1

        if exit_flag:
            break

        amount_remaining -= 1

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

    if marks:  # Check if marks dictionary is not empty
        print("\nMarksheet:")
        print("Letter | Result    | Time Taken")
        for letter, (result, time_taken) in sorted(marks.items()):
            print(f"{letter}      | {result} | {time_taken} seconds")
        total_correct = sum(1 for result, _ in marks.values() if result == "Correct")
        final_score = round((total_correct / len(marks)) * 100, 2)
        print(f"Final Score: {final_score}% (Correct: {total_correct} out of {len(marks)})")
    else:
        print("No attempts were made.")
    # Save final progress
    save_progress(progress, file_path)

def main():
    SCRIPT_DIR, DATA_DIR = get_directory()
    model = open_model(SCRIPT_DIR, DATA_DIR)
    settings = practice_settings()
    progress_file = "user_progress.pkl"
    user_progress = load_progress(progress_file)
    practice_loop(model, user_progress, progress_file, settings)

if __name__ == "__main__":
    main()