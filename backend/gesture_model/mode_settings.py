"""
This program hold allows users to save their progress, and settings.
This holds the information for both the quiz and practice mode settings.
"""
import pickle
import os
import cv2


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

    return user_progress


# Saving the users progress from their practices. 
def save_progress(progress, file_path) -> None:
    with open(file_path, 'wb') as file:
        pickle.dump(progress, file)

def practice_settings():
    default_settings = {"Time for each letter (seconds)":5,
                        "Amount of letters to practice":5
                        }
    print(default_settings)
    change_settings = input("Would you like to change any of the default settings? (y/n): ").strip().lower()
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


def display_settings():
    # Define colors for the boxes and text
    box_color = (255, 255, 255)  # White color for boxes
    text_color = (0, 0, 0)  # Black color for text
    correct_color = (0, 255, 0)  # Green color for correct feedback
    incorrect_color = (0, 0, 255)  # Red color for incorrect feedback
    font = cv2.FONT_HERSHEY_SIMPLEX
    display_correct = True

    return box_color, text_color, correct_color, incorrect_color, font, display_correct