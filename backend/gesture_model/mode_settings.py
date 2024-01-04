"""
This program hold allows users to save their progress, and settings.
This holds the information for both the quiz and practice mode settings.
"""
import pickle
import os


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
    with open(file_path, 'wb') as file:
        pickle.dump(progress, file)

def practice_settings():
    default_settings = {"Time for each letter (seconds):":5,
                        "Amount of letters to practice":5
                        }
    print(default_settings)
    change_settings = input("Would you like to change any of the default settings? (y/n)").strip().lower()
    if change_settings == "y":
        while True:
            try:
                amount_of_letters = int(input("How many letters would you like to practice this session?").lower().strip())
                default_settings["Amount of letters to practice"] = amount_of_letters
                break
            except:
                print("Try again. Please only submit integer values.")      
        while True:
            try:
                time_wanted = int(input("How much time would you like for each letter?").lower().strip())
                default_settings["Time for each letter (seconds):"] = time_wanted
                break
            except:
                print("Try again. Please only submit integer values.")      
    return default_settings