"""
This program hold allows users to save their progress, and settings.
This holds the information for both the quiz and practice mode settings.
"""
import pickle
import os


# Loading the users progress from their practices. 
def load_progress(progress_file: str) -> dict:
    if os.path.exists(progress_file):
        reset_choice = input("You have previously existing progress.\nDo you want to reset your progress or view previous marks?\nEnter 'r' to reset, 'v' to view, or any other key to continue: ").strip().lower()
        if reset_choice == 'r':
            user_progress = {chr(65+i): {'attempts': 0, 'correct': 0, 'times': []} for i in range(26)}
            save_progress(user_progress, progress_file)
            print("Progress has been reset.")
        elif reset_choice == 'v':
            with open(progress_file, 'rb') as file:
                user_progress = pickle.load(file)
            print("Previous Marks:")
            for letter, stats in user_progress.items():
                print(f"Letter: {letter}, Attempts: {stats['attempts']}, Correct: {stats['correct']}, Times: {stats['times']}")
            user_progress = load_progress(progress_file)  # Reload to either reset or continue
        else:
            with open(progress_file, 'rb') as file:
                user_progress = pickle.load(file)
            print("Continuing with existing progress.")
    else:
        user_progress = {chr(65+i): {'attempts': 0, 'correct': 0, 'times': []} for i in range(26)}

    if input("Would you like to see your progress? (y/n)").lower().strip() == "y":
        print(user_progress)
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
                amount_of_letters = int(input("How many letters would you like to practice this session? :").lower().strip())
                default_settings["Amount of letters to practice"] = amount_of_letters
                break
            except:
                print("Try again. Please only submit integer values.")      
        while True:
            try:
                time_wanted = int(input("How much time would you like for each letter? :").lower().strip())
                default_settings["Time for each letter (seconds)"] = time_wanted
                break
            except:
                print("Try again. Please only submit integer values.")      
    return default_settings