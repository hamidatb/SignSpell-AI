import cv2
import mediapipe as mp
import os
import numpy as np
import time
import sys
import random
from hand_gesture_recognizer import recognize_letter
from test_classifier import open_model
from mode_settings import load_progress, practice_settings, save_letter_quiz, save_word_quiz
from mode_settings import display_settings, quiz_words


def get_directory() -> str:
    # Getting the directory where this script file is located
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    MODEL_DIR = os.path.join(SCRIPT_DIR, "model.p")

    # Go up one directory level to the 'backend' directory
    BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
    # Now set the path to the 'static' directory
    IMAGES_DIR = os.path.join(BACKEND_DIR, "static")
        
    return SCRIPT_DIR, MODEL_DIR, IMAGES_DIR

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

def get_letter_image(images_dir, target_letter):
    letter_image_path = os.path.join(images_dir, f"{target_letter.upper()}.png")
    letter_image = cv2.imread(letter_image_path)

    return letter_image

def update_and_display(frame, target_letter, predicted_character, amount_remaining, time_remaining, images_dir):
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

# -> Need to update this function to work
def select_quiz_letter(progress):
    """
    Selecting the next letter for practice using a simple heuristic:
    Less accuracy and fewer attempts increase the likelihood of selection.
    """
    letters = list(progress.keys())
    # Calculate weights inversely proportional to accuracy and attempts
    weights = [1 / (progress[letter]['correct'] + 0.1) / (progress[letter]['attempts'] + 1) for letter in letters]
    return random.choices(letters, weights)[0]

# -> This function isn't actually written yet lol
def select_quiz_word(progress):
    pass

# Item 2. Saturday
def quiz_letters(model,letter_quiz_file, settings, images_dir):
    cap, hands = initialize_camera()
    total_attempts = 0
    total_correct = 0

    # Intializing the scorescheet dictionary for all of the letters.
    letter_accuracies = {chr(65 + i): {'attempts': 0, 'correct': 0} for i in range(26)}
    
    # Using a try, except, finally block to execute the the quiz letters logic.
    try:
        for i in range(settings["Amount of letters to practice"]):
            target_letter = select_quiz_letter(letter_quiz_file)
            print(f"Target letter to display: {target_letter}")
            start_time = time.time()
            letter_accuracies[target_letter]['attempts'] += 1
            total_attempts += 1
            
            while True:
                frame, results = capture_and_process_frame(cap, hands)
                predicted_character = make_prediction(model, results, frame)

                # You will break if they show the correct character
                if predicted_character and predicted_character.lower() == target_letter.lower():
                    letter_accuracies[target_letter]['correct'] += 1
                    total_correct += 1
                    break  # Move to next letter immediately upon correct predicti

                if (time.time() - start_time) > settings["Time for each letter (seconds)"]:
                    break  # Move to next letter if time limit is exceeded

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt  # Use an exception to break out of the loop
            
            time.sleep(0.5)  # Short pause between letters (optional)
    
    except KeyboardInterrupt:
        print("Quiz ended early by the user.")

    # The finally block executes after the try and except block regardless of what happens!
    finally:
        # Closing the mediapipe cv2 windows
        cap.release()
        cv2.destroyAllWindows()
        
        # letter accuracies is essentially just the quiz maks
        print("Quiz Results:")
        for letter, stats in letter_accuracies.items():
            if stats['attempts'] > 0:
                accuracy = (stats['correct'] / stats['attempts']) * 100
                print(f"Letter {letter}: {accuracy:.2f}% accuracy")
        
        overall_accuracy = (total_correct / total_attempts) * 100 if total_attempts > 0 else 0
        print(f"Overall accuracy: {overall_accuracy:.2f}%")

        save_letter_quiz(letter_accuracies)
    
    print(letter_accuracies)


# Item 1. Saturday
def quiz_words(model, progress, file_path, settings, images_dir):
    cap, hands = initialize_camera()
    # initialize the tracking of the score of this quiz

    # Things to ask before starting:
    # 1. Would you like to view or reset your previous quiz marks?
    # 2. Show the deault settings (3 words, 3 mins max with the ability to self end the quiz)
    #    Ask if they'd like to view or change the default settings. 
    #    2.a) How many words would you like to be quizzed on today (int)
    #    2.b) What is the maximum time you'd like for this word quiz (int)
    # 3. Use the same ANKI algorithmn to decided which words they will be quizzed on in order.


    # The main quiz logic:
    # (For each word in the range of the amount of words the user chose, this is what will happen,)
    # (note: You need to keep track of the start and end time for each word and put in)
    # 1. Show the letters of the word they're being quizzed on in red. As the ML algroithmr recognizes each 
    #   letter, turn that letter green. Once all the letters are green, consider that word done and record the time
    #   that it took on the marksheet. If any specific letter took a long time, record it in a systematic way
    #   in the notes section of the marksheet.
    # 2. They also need to be able to see a countdown timer on their screen.
    # 3. They also need to see the mediapipe box for the hand gesture recognition on their screen as well.
    # 4. Return the marksheet, the progress dict with this entry added (acompnaied with the date and time it was done), and the feedback marksheet stuff as a srting.
    # 5. Add visuals to make this a very clean and good game.

    # The quiz_letters duntion should work similarly but only quiz letters.

# Item 3. Saturday
def save_quiz_progress():
    pass

# Item 4. Saturday
def load_quiz_progress():
    pass


def main():
    SCRIPT_DIR, MODEL_DIR, IMAGES_DIR = get_directory()
    model = open_model(SCRIPT_DIR, MODEL_DIR)
    settings = practice_settings()
    progress_file = "user_progress.pkl"
    letter_quiz_marks = load_letter_quiz(letter_progress_file)
    word_quiz_marks = load_word_quiz(word_progess_file)

if __name__ == "__main__":
    main()