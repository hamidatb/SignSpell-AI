import cv2
import mediapipe as mp
import os
import numpy as np
import time
import sys
import random
import base64
from .test_classifier import open_model
from .mode_settings import save_letter_quiz, save_word_quiz
from .mode_settings import display_settings, present_user_options_for_marks, letter_quiz_settings, word_quiz_settings

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from socketio_setup import socketio  # Absolute import

def emit_terminal_output(output):
    socketio.emit('terminal_output', {'output': output})

def emit_question(question, options):
    socketio.emit('quiz_question', {'question': question, 'options': options})

def emit_frame(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    frame_encoded = base64.b64encode(buffer).decode('utf-8')
    socketio.emit('video_frame', {'frame': frame_encoded})

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
        emit_terminal_output("Failed to grab frame, retrying...")
        return None, None  # Instead of exiting, return None to indicate failure
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

def update_and_display(frame, target_letter, predicted_character, amount_remaining, time_remaining, images_dir):
    # Define colors for the boxes and text
    box_color = (255, 255, 255)  # White color for boxes
    text_color = (0, 0, 0)  # Black color for text
    correct_color = (0, 255, 0)  # Green color for correct feedback
    incorrect_color = (0, 0, 255)  # Red color for incorrect feedback
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Check if a prediction was made
    if predicted_character is not None:
        is_correct = predicted_character.lower() == target_letter.lower()
        feedback_text = "Correct!" if is_correct else "Incorrect!"
        feedback_color = correct_color if is_correct else incorrect_color

        # Display feedback for a correct prediction
        if is_correct:
            # Fill the rectangle with green color
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), correct_color, cv2.FILLED)
            cv2.putText(frame, feedback_text, (frame.shape[1] // 2 - 100, frame.shape[0] // 2),
                        font, 1, text_color, 2)
            emit_frame(frame)
            socketio.sleep(1)  # Wait for 1 second
            return is_correct
    else:
        is_correct = False
        feedback_text = "No prediction"
        feedback_color = incorrect_color

    # Put text on the frame
    cv2.putText(frame, f"Show this letter: {target_letter}", (10, 80), font, 0.7, text_color, 2)
    cv2.putText(frame, feedback_text, (300, 80), font, 0.7, feedback_color, 2)
    
    # Display remaining amount and time at the bottom
    cv2.putText(frame, f"Amount remaining: {amount_remaining}", (10, frame.shape[0] - 50), font, 0.7, text_color, 2)
    cv2.putText(frame, f"Seconds remaining: {time_remaining:.2f}s", (300, frame.shape[0] - 50), font, 0.7, text_color, 2)

    # Display instruction on the bottom right
    quit_text = "Press 'q' to quit"
    text_width, _ = cv2.getTextSize(quit_text, font, 0.7, 2)[0]
    cv2.putText(frame, quit_text, (frame.shape[1] - text_width - 10, frame.shape[0] - 20), font, 0.7, text_color, 2)
    
    emit_frame(frame)
    return is_correct

def update_and_display_word(frame, target_word, current_index, predicted_character, time_remaining, images_dir):
    box_color = (255, 255, 255)  # White color for boxes
    text_color = (0, 0, 0)       # Black color for text
    correct_color = (0, 255, 0)  # Green color for correct feedback
    incorrect_color = (0, 0, 255)# Red color for incorrect feedback
    font = cv2.FONT_HERSHEY_SIMPLEX
    letter_colors = []

    # Initialize letter colors based on current progress
    for i in range(len(target_word)):
        if i < current_index:
            # Letters before the current index are considered correct
            letter_colors.append(correct_color)
        elif i == current_index and predicted_character and predicted_character.lower() == target_word[i].lower():
            # Current letter is correct
            letter_colors.append(correct_color)
        else:
            # Remaining letters or incorrect current letter
            letter_colors.append(incorrect_color)

    # Display the word with colored letters
    for i, letter in enumerate(target_word):
        letter_pos_x = 50 + i * 20  # Horizontal position of the letter
        # cv2.putText(frame, text to show, coordinate, font, fotn size, font colour, font boldness)
        cv2.putText(frame, letter.upper(), (letter_pos_x, 80), font, 1, letter_colors[i], 2)

    # Display remaining time
    cv2.putText(frame, f"Seconds remaining: {time_remaining:.2f}s", (10, frame.shape[0] - 20), font, 0.7, text_color, 2)

    # Display instruction on the bottom right
    quit_text = "Spell the words using the fingerspell alphabet. Press 'q' to quit"
    text_width, _ = cv2.getTextSize(quit_text, font, 0.7, 2)[0]
    cv2.putText(frame, quit_text, (frame.shape[1] - text_width - 10, frame.shape[0] - 20), font, 0.7, text_color, 2)

    emit_frame(frame)

    # Return True if the current letter is correct, False otherwise
    return current_index < len(target_word) and predicted_character and predicted_character.lower() == target_word[current_index].lower()

def select_quiz_letter(progress):
    """
    Selecting the next letter for practice using a simple heuristic:
    Less accuracy and fewer attempts increase the likelihood of selection.
    """
    letters = list(progress.keys())
    # Calculate weights inversely proportional to accuracy and attempts
    weights = [1 / (progress[letter]['correct'] + 0.1) / (progress[letter]['attempts'] + 1) for letter in letters]
    return random.choices(letters, weights)[0]

def quiz_letters(model, letter_quiz_marks, letter_quiz_settings, images_dir):
    global quiz_running
    cap, hands = initialize_camera()
    
    total_attempts = 0
    total_correct = 0

    # Initialize the score sheet dictionary for all of the letters
    letter_accuracies = {chr(65 + i): {'attempts': 0, 'correct': 0} for i in range(26)}
    print("Initialized accuracies")
    try:
        print("Failed at try")
        for i in range(letter_quiz_settings["Amount of letters to be quizzed on"]):
            target_letter = select_quiz_letter(letter_quiz_marks)
            emit_terminal_output(f"Target letter to display: {target_letter}")
            start_time = time.time()
            letter_accuracies[target_letter]['attempts'] += 1
            total_attempts += 1
            print("Made it to total attempts")

            # A flag to control the outer loop (So the user can quit by pressing 'q')
            exit_flag = False
            print("Made it to exit flag")
            amount_remaining = letter_quiz_settings["Amount of letters to be quizzed on"] - i - 1
            print("Made it to amount remaining")
            while True:
                # Process frame and make predictions
                frame, results = capture_and_process_frame(cap, hands)
                if frame is None:
                    continue  # Retry if frame capture failed
                print("Right before predicitng the char")
                predicted_character = make_prediction(model, results, frame)
                print("Right before time remaining")
                time_remaining = letter_quiz_settings["Time for each letter (seconds)"] - (time.time() - start_time)
                print(f"{time_remaining}")
                # Display updates: This is where it currently breaks
                is_correct = update_and_display(frame, target_letter, predicted_character, amount_remaining, time_remaining, images_dir)
                print("got to is correct")
                print(f"{is_correct}")
                # Check if the prediction is correct
                if is_correct:
                    letter_accuracies[target_letter]['correct'] += 1
                    total_correct += 1
                    cv2.waitKey(1000)  # Display Correct! for 1 second
                    break
                else:
                    print("Was not correct")

                if time_remaining <= 0:
                    break  # Exit the loop if the time is up
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    exit_flag = True
                    break
            
            if exit_flag:
                emit_terminal_output("Quiz ended early by the user.")
                break
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        # Calculate and print the quiz results
        emit_terminal_output("Quiz Results:")
        for letter, stats in letter_accuracies.items():
            if stats['attempts'] > 0:
                accuracy = (stats['correct'] / stats['attempts']) * 100
                emit_terminal_output(f"Letter {letter}: {accuracy:.2f}% accuracy")
        
        overall_accuracy = (total_correct / total_attempts) * 100 if total_attempts > 0 else 0
        emit_terminal_output(f"Overall accuracy: {overall_accuracy:.2f}%")

        # Save the quiz marks
        save_letter_quiz(letter_accuracies, "update marks")
        quiz_running = False
    
def quiz_words(model, word_quiz_marks, word_quiz_settings, images_dir):
    cap, hands = initialize_camera()

    total_attempts = 0
    total_correct_words = 0
    word_accuracies = {}  # Dictionary to store attempts and correct counts for each word
    used_words = []

    try:
        for i in range(word_quiz_settings["Amount of words to be quizzed on"]):
            target_word = select_quiz_word(used_words)  # Function to select a word
            print(f"Word to display: {target_word}")
            start_time = time.time()
            word_accuracies[target_word] = {'attempts': 0, 'correct': 0}
            word_accuracies[target_word]['attempts'] += 1
            total_attempts += 1

            current_index = 0  # Current index of the letter in the word

            while current_index < len(target_word):
                frame, results = capture_and_process_frame(cap, hands)
                predicted_character = make_prediction(model, results, frame)
                time_remaining = word_quiz_settings["Time for each word (seconds)"] - (time.time() - start_time)

                # Display the word and check the prediction
                is_correct = update_and_display_word(frame, target_word, current_index, predicted_character, time_remaining, images_dir)

                if is_correct:
                    current_index += 1  # Move to the next letter if correct

                # Allowing the user to press q to break out of the program by pressing q to force an exit.
                if time_remaining <= 0 or cv2.waitKey(1) & 0xFF == ord('q'):
                    break  # Exit if time's up or 'q' is pressed

            if current_index == len(target_word):
                word_accuracies[target_word]['correct'] += 1
                total_correct_words += 1

            time.sleep(0.5)  # Short pause between words

    except KeyboardInterrupt:
        print("Quiz ended early by the user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

        # Calculate and display the quiz results
        print("Word Quiz Results:\n")
        for word, stats in word_accuracies.items():
            accuracy = (stats['correct'] / stats['attempts']) * 100 if stats['attempts'] > 0 else 0
            print(f"Word {word}: {accuracy:.2f}% accuracy")
        
        overall_accuracy = (total_correct_words / total_attempts) * 100 if total_attempts > 0 else 0
        print(f"\nOverall word accuracy: {overall_accuracy:.2f}%")

        save_word_quiz(word_accuracies, "word_quiz_results.pkl")  # Save the quiz results
        return word_accuracies

def select_quiz_word(used_words):
    # List of 10 common ASL words
    common_asl_words = ["hello", "sorry", "please", "thank", "help", "love", "yes", "no", "friend", "family"]
    available_words = [word for word in common_asl_words if word not in used_words]

    word_choice = random.choice(available_words) if available_words else None
    used_words.append(word_choice)

    return word_choice

def type_of_quiz():
    question = "What type of quiz would you like to do?"
    options = "Options: Letter quiz, word quiz, quit. (input l, w, or q): "
    emit_question(question, options)
    socketio.sleep(0.1)

@socketio.on('quiz_answer')
def handle_quiz_answer(data):
    choice = data['answer'].strip().lower()
    if choice not in ("l", "w", "q"):
        print(f"User entered invalid choice: {choice}")
        emit_terminal_output("Please respond with either 'l', 'w', or 'q'")
    else:
        SCRIPT_DIR, MODEL_DIR, IMAGES_DIR = get_directory()
        model = open_model(SCRIPT_DIR, MODEL_DIR)

        socketio.emit('quiz_choice', {'choice': choice})
        print(f"{choice}")

        if choice == "l":
            l_quiz_settings = letter_quiz_settings()
            letter_quiz_marks =  present_user_options_for_marks(choice)
            
            if letter_quiz_marks == None:
                letter_quiz_marks = save_letter_quiz(None, "reset marks") # returns an empty dict of marks that have been saved to a file.
            
            quiz_letters(model, letter_quiz_marks, l_quiz_settings, IMAGES_DIR)

        elif choice == "w":
            w_quiz_settings = word_quiz_settings()
            word_quiz_marks =  present_user_options_for_marks(choice)
            
            if word_quiz_marks == None:
                word_quiz_marks = save_letter_quiz(None, "reset marks") # returns an empty dict of marks that have been saved to a file.
            
            quiz_words(model, word_quiz_marks,w_quiz_settings, IMAGES_DIR)

        elif choice == "q":
            emit_terminal_output("Thank you for trying SignSpell!")
            return

    
def quiz_main():
    global quiz_running
    SCRIPT_DIR, MODEL_DIR, IMAGES_DIR = get_directory()
    model = open_model(SCRIPT_DIR, MODEL_DIR)

    try:
        type_of_quiz()
    finally:
        quiz_running = False  # Ensure quiz_running is set to False when the quiz ends

if __name__ == "__main__":
    quiz_main()


