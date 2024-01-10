"""
This program hold allows users to save their progress, and settings.
This holds the information for both the quiz and practice mode settings.
"""
import pickle
import os
import cv2

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
    print(f"\n")
    for key, value in default_settings.items():
        print(f"{key}:{value}")
    print(f"\n")
    change_settings = input("Would you like to change any of the default settings? (y/n): ").strip().lower()
    print(f"\n")
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

    if type_of_save == "reset_marks":
        letter_accuracies = {chr(65 + i): {'attempts': 0, 'correct': 0} for i in range(26)}
    
    with open(letter_quiz_file_path, "wb") as file:
            pickle.dump(letter_accuracies, file)    
    
    print('Your letter quiz marks have been updated.')

def save_word_quiz(word_quiz_marks):

    run_data_dir = get_run_data_file_path()

    word_quiz_file_path = os.path.join(run_data_dir, "word_quiz_marks.pkl")

    with open(word_quiz_file_path, "wb") as file:
        pickle.dump(word_quiz_marks,file)

    print(f"Your finger spell word quiz marks have been saved to {word_quiz_file_path}")


def present_user_options_for_marks(type_of_quiz:str):
    run_data_dir = get_run_data_file_path()

    if type_of_quiz == "l":
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
