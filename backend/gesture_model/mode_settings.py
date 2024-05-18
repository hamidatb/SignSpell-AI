"""
This program hold allows users to save their progress, and settings.
This holds the information for both the quiz and practice mode settings.
"""
import pickle
import os
import cv2
import sys
import base64
from socketio_setup import socketio  # Absolute import

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)


def emit_terminal_output(output):
    socketio.emit('terminal_output', {'output': output})

def emit_question(question, options):
    socketio.emit('quiz_question', {'question': question, 'options': options})

def emit_frame(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    frame_encoded = base64.b64encode(buffer).decode('utf-8')
    socketio.emit('video_frame', {'frame': frame_encoded})


def display_settings():
    # Define colors for the boxes and text
    box_color = (255, 255, 255)  # White color for boxes
    text_color = (0, 0, 0)  # Black color for text
    correct_color = (0, 255, 0)  # Green color for correct feedback
    incorrect_color = (0, 0, 255)  # Red color for incorrect feedback
    font = cv2.FONT_HERSHEY_SIMPLEX
    display_correct = True

    return box_color, text_color, correct_color, incorrect_color, font, display_correct

def practice_settings():
    default_settings = {"Time for each letter (seconds)":5,
                        "Amount of letters to practice":5}
    emit_terminal_output(f"\n")
    for key, value in default_settings.items():
        emit_terminal_output(f"{key}:{value}")
    emit_terminal_output(f"\n")
    emit_terminal_output("Would you like to change any of the default settings? (y/n): ")
    change_settings = input("").strip().lower()
    emit_terminal_output(f"\n")
    if change_settings == "y":
        while True:
            try:
                amount_of_letters = int(input("How many letters would you like to practice this session? : ").lower().strip())
                default_settings["Amount of letters to practice"] = amount_of_letters
                break
            except:
                print("Try again. Please only submit integer values.")      
        while True:
            try:
                time_wanted = int(input("How much time would you like for each letter? : ").lower().strip())
                default_settings["Time for each letter (seconds)"] = time_wanted
                break
            except:
                print("Try again. Please only submit integer values.")      
    return default_settings

def letter_quiz_settings():
    default_settings = {"Time for each letter (seconds)":5,
                        "Amount of letters to be quizzed on":5}
    emit_terminal_output(f"\n")
    for key, value in default_settings.items():
        emit_terminal_output(f"{key}: {value}\n")
    emit_terminal_output("Would you like to change any of the default quiz settings? (y/n):")
    change_settings = input("Would you like to change any of the default quiz settings? (y/n): ").strip().lower()
    emit_terminal_output(f"\n")
    if change_settings == "y":
        while True:
            try:
                amount_of_letters = int(input("How many letters would you like to be quizzed on this session? : ").lower().strip())
                default_settings["Amount of letters to be quizzed on"] = amount_of_letters
                break
            except:
                print("Try again. Please only submit integer values.")      
        while True:
            try:
                time_wanted = int(input("How much time would you like for each letter? : ").lower().strip())
                default_settings["Time for each letter (seconds)"] = time_wanted
                break
            except:
                print("Try again. Please only submit integer values.")      
    return default_settings

def word_quiz_settings():
    default_settings = {
        "Amount of words to be quizzed on": 3,
        "Time for each word (seconds)": 180  # 3 minutes as default
    }

    print("\nCurrent word quiz settings:")
    for key, value in default_settings.items():
        print(f"{key}: {value}")

    if input("Would you like to change any of the default settings? (y/n): ").strip().lower() == "y":
        try:
            num_words = int(input("How many words would you like to be quizzed on? : ").strip())
            default_settings["Amount of words to be quizzed on"] = num_words

            max_time_per_word = int(input("Maximum time per word (in seconds): ").strip())
            default_settings["Time for each word (seconds)"] = max_time_per_word
        except ValueError:
            print("Invalid input. Using default settings.")

    return default_settings 

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_run_data_file_path() -> str:
    """ This gets the current script dir, gets the run script directory and ensures that it exists. 
    It returns a file path as a stirng."""
    current_script_dir = os.path.dirname(os.path.realpath(__file__))
    run_script_dir = os.path.join(current_script_dir, "run_data")

    ensure_directory_exists(run_script_dir)

    return run_script_dir

# Loading the users progress from their practices. 
def load_progress(progress_file: str) -> dict:
    if os.path.exists(progress_file):
        with open(progress_file, 'rb') as file:
            user_progress = pickle.load(file)

        reset_choice = input("You have previously existing progress.\nDo you want to reset your progress, view previous marks, or continue with existing marks?\nEnter 'r' to reset, 'v' to view, or any other key to continue: ").strip().lower()
        if reset_choice == 'r':
            user_progress = {chr(65 + i): {'attempts': 0, 'correct': 0, 'times': []} for i in range(26)}
            save_progress(user_progress, progress_file)
            print("Progress has been reset.")
        elif reset_choice == 'v':
            print("Previous Marks:")
            for letter, stats in user_progress.items():
                # Ensure 'times' key exists
                stats['times'] = stats.get('times', [])
                print(f"Letter: {letter}, Attempts: {stats['attempts']}, Correct: {stats['correct']}, Times: {stats['times']}")
            user_progress = load_progress(progress_file)  # Reload to either reset or continue
        else:
            print("Continuing with existing progress.")
            # Ensure 'times' key exists for each letter
            for letter in user_progress:
                if 'times' not in user_progress[letter]:
                    user_progress[letter]['times'] = []
            save_progress(user_progress, progress_file)
    else:
        user_progress = {chr(65 + i): {'attempts': 0, 'correct': 0, 'times': []} for i in range(26)}
    print(f"\n")
    return user_progress

# Saving the users progress from their practices. 
def save_progress(progress, file_path) -> None:
    with open(file_path, 'wb') as file:
        pickle.dump(progress, file)

def save_letter_quiz(letter_accuracies, type_of_save):
    run_data_dir = get_run_data_file_path()
    letter_quiz_file_path = os.path.join(run_data_dir, "letter_quiz_marks.pkl")

    if type_of_save == "reset marks" or letter_accuracies == None:
        letter_accuracies = {chr(65 + i): {'attempts': 0, 'correct': 0} for i in range(26)}
    
    with open(letter_quiz_file_path, "wb") as file:
            pickle.dump(letter_accuracies, file)    
    
    print('Your letter quiz marks have been updated.')
    return letter_accuracies

def save_word_quiz(word_quiz_marks, type_of_save):
    run_data_dir = get_run_data_file_path()
    word_quiz_file_path = os.path.join(run_data_dir, "word_quiz_marks.pkl")

    # Reset the marks if requested
    if type_of_save == "reset_marks" or word_quiz_marks == None:
        word_quiz_marks = {}  # Reset to an empty dictionary

    # Save the word quiz marks to the file
    with open(word_quiz_file_path, "wb") as file:
        pickle.dump(word_quiz_marks, file)

    print(f"Word quiz marks have been {'reset' if type_of_save == 'reset_marks' else 'saved'} in '{word_quiz_file_path}'.")
    return word_quiz_marks

def present_user_options_for_marks(type_of_quiz:str):
    run_data_dir = get_run_data_file_path()

    if type_of_quiz in ("l", "letter"):
        type_of_quiz = "letter"
    else:
        type_of_quiz = "word"

    file_path = os.path.join(run_data_dir, f"{type_of_quiz}_quiz_marks.pkl")

    if not os.path.exists(file_path):
        print(f"No previous quiz data found.")
        return None

    # Present the options to the user
    print(f"Previous {type_of_quiz} quiz marks found. Choose an option:")
    print("1. View past marks")
    print("2. Clear past marks")
    print("3. Continue with existing marks")
    user_choice = input("Enter the number of your choice (1/2/3): ").strip()

    if user_choice == '1':
        # View past marks
        with open(file_path, 'rb') as file:
            quiz_marks = pickle.load(file)
        print("Past marks:")
        for key, value in quiz_marks.items():
            print(f"{key}: {value}")
        return quiz_marks

    # Need to make 2 the same as if you just started a new file - essentially reset.
    elif user_choice == '2':
        # Clear past marks by removing the file
        os.remove(file_path)
        
        return None

    elif user_choice == '3':
        # Continue with existing marks by loading them
        with open(file_path, 'rb') as file:
            quiz_marks = pickle.load(file)
        return quiz_marks

    else:
        print("Invalid option selected. No action taken.")
        return None
    
    # If this is equal to none, you have to use the save file to create a new file since the user didn't want the old file.
