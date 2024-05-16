import cv2
import mediapipe as mp
import os
import numpy as np
import time
import sys
import random
from .test_classifier import open_model
from .mode_settings import load_progress, save_progress, practice_settings
from .mode_settings import display_settings


def get_directory() -> str:
    # Getting the directory where this script file is located
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    MODEL_DIR = os.path.join(SCRIPT_DIR, "model.p")

    # Go up one directory level to the 'backend' directory
    BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
    # Now set the path to the 'static' directory
    IMAGES_DIR = os.path.join(BACKEND_DIR, "static")
        
    return SCRIPT_DIR, MODEL_DIR, IMAGES_DIR

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
    x_ = []
    y_ = []

    height, width, _  = frame.shape
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                data_loc.extend([x, y])

                # These lists are being made to determine the bounding box limits
                x_.append(x)
                y_.append(y)


        prediction = model.predict([np.asarray(data_loc)])
        predicted_character = chr(65 + int(prediction[0]))  
        return predicted_character, x_, y_, height, width
    return None, None, None, None, None

def get_letter_image(images_dir, target_letter):
    letter_image_path = os.path.join(images_dir, f"{target_letter.upper()}.png")
    letter_image = cv2.imread(letter_image_path)

    return letter_image

def update_and_display(frame, target_letter, predicted_character, amount_remaining, time_remaining, images_dir, x_, y_, height, width):
    box_color, text_color, correct_color, incorrect_color, font, display_correct = display_settings()

    # Drawing the boxes for visual design
    cv2.rectangle(frame, (10, 40), (300, 100), box_color, 2)  # Box for 'Show this letter'
    cv2.rectangle(frame, (400, 40), (690, 100), box_color, 2)  # Box for 'Accuracy'
    
    if get_letter_image(images_dir, target_letter) is not None:
        # Resize image if necessary and put it on the frame
        letter_image = cv2.resize(get_letter_image(images_dir, target_letter), (100, 100))
        frame[100:200, 100:200] = letter_image  # Adjust position as needed

    # Check if a prediction was made
    if predicted_character is not None:
         # Increasing the margin for the bounding box
        margin_x = 20  
        margin_y = 20  

        x1, y1 = max(int(min(x_) * width) - margin_x, 0), max(int(min(y_) * height) - margin_y, 0)
        x2, y2 = min(int(max(x_) * width) + margin_x, width), min(int(max(y_) * height) + margin_y, height)

        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,0), 4)
        cv2.putText(frame, predicted_character, (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2, cv2.LINE_AA)

        is_correct = predicted_character.lower() == target_letter.lower()
        feedback_color = correct_color if is_correct else incorrect_color
        if display_correct and is_correct:
            # If the prediction is correct and we want to display the "Correct!" feedback:
            cv2.rectangle(frame, (400, 40), (690, 100), correct_color, -1)  # Draw a filled green rectangle
            feedback_text = "Correct!"
            cv2.putText(frame, feedback_text, (450, 80), font, 0.7, text_color, 2)
        else:
            feedback_text = f"Accuracy: {'Correct!' if is_correct else 'Incorrect!'}"
            cv2.putText(frame, feedback_text, (450, 80), font, 0.7, feedback_color, 2)
    else:
        is_correct = False
        feedback_text = "Accuracy: No prediction"
        feedback_color = incorrect_color

    # Put text on the frame
    cv2.putText(frame, f"Show this letter: {target_letter}", (60, 80), font, 0.7, text_color, 2)
    cv2.putText(frame, feedback_text, (450, 80), font, 0.7, feedback_color, 2)
    
    # Display remaining amount and time at the bottom
    cv2.putText(frame, f"Amount remaining: {amount_remaining}", (10, frame.shape[0] - 50), font, 0.7, text_color, 2)
    cv2.putText(frame, f"Seconds remaining: {time_remaining:.2f}s", (10, frame.shape[0] - 20), font, 0.7, text_color, 2)

    bottom_middle_text = "Press 'q' to end your session early"
    text_width, _ = cv2.getTextSize(bottom_middle_text, font, 0.7, 2)[0]
    cv2.putText(frame, bottom_middle_text, ((frame.shape[1] - text_width) // 2, frame.shape[0] - 20), font, 0.7, text_color, 2)

    cv2.imshow("Practice Mode", frame)

    return is_correct

def practice_loop(model, progress, file_path, settings, images_dir):
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
            predicted_character, x_, y_, height, width = make_prediction(model, results, frame)
            is_correct = update_and_display(frame, target_letter, predicted_character, amount_remaining, time_remaining, images_dir, x_, y_, height, width)

            key = cv2.waitKey(1)
            if key == ord('q'):
                exit_flag = True
                break

            end_time = time.time()
            time_taken = round(end_time - start_time, 2)
            
            if is_correct:
                # If the prediction is correct, display the green "Correct!" feedback for 1 second
                update_and_display(frame, target_letter, predicted_character, amount_remaining, time_remaining, images_dir, x_, y_, height, width)
                
    
                progress[target_letter]['correct'] += 1
                progress[target_letter]['times'].append(time_taken)
                marks[target_letter] = ("Correct", time_taken)

                cv2.waitKey(1000)  # Wait for 1s

                break  # Then break out of the loop to move on to the next letter
            else:
                marks[target_letter] = ("Incorrect", time_taken)

            progress[target_letter]['attempts'] += 1


        if exit_flag:
            print("Exiting the practice loop.")
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
        print(f"Final Score: {final_score}% (Correct: {total_correct} out of {len(marks)})\n")
    else:
        print("No full attempts were made.")
    # Save final progress
    save_progress(progress, file_path)

    print(marks)
    return marks, progress

def main():
    SCRIPT_DIR, MODEL_DIR, IMAGES_DIR = get_directory()
    model = open_model(SCRIPT_DIR, MODEL_DIR)
    settings = practice_settings()
    progress_file = "user_progress.pkl"
    user_progress = load_progress(progress_file)
    practice_mark, practice_progress = practice_loop(model, user_progress, progress_file, settings, IMAGES_DIR)
    
    return practice_mark, practice_progress

if __name__ == "__main__":
    main()