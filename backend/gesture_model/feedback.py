import os
import sys
from openai import OpenAI
from .practice_mode import get_directory
from dotenv import load_dotenv

# Loading the envirnoment variables from the .env file (API key)
# Note: Only I have access to my specific .env file. If you'd like to run this part, copythis repo to your device and add your own .env file.
load_dotenv()

try:
    api_key = os.getenv('SIGNSPELL_API_KEY')
except ValueError:
    raise ValueError("No OPENAI_API_KEY found in environment variables")

client = OpenAI(api_key=api_key)

# The main function interacting with the openAI API
def ask_gpt(task_needed, marks, mark_history):
    gpt_role = "You are a kind and friendly American Sign Language Finger Spelling teacher. You want to give supportive feedback based on marks an errors, or just enouragement. Users are trying to use the program to learn ASL, and you're the kind feedback"

    if marks and mark_history:
        completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system",
        "content": f"{gpt_role}"
        },

        {"role": "user",
        "content": f"{task_needed} these are their marks on this attempt:{marks}, these were their previous marks {mark_history}"
        }
    ]
    )
    elif marks:
        completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
            "content": f"{gpt_role}"
            },

            {"role": "user",
        "content": f"{task_needed} these are their marks on this attempt:{marks}"
            }
        ]
        )
    else:
        completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
            "content": f"{gpt_role}"
            },

            {"role": "user",
        "content": f"{task_needed}"
            }
        ]
        )

    raw_message = str(completion.choices[0].message)
    formatted_message = raw_message.replace('\\n', '\n')
    print(raw_message, formatted_message)

    # Split the string to extract the message part
    message_start = formatted_message.find('content="') + len('content="')
    message_end = formatted_message.find('", role=')

    # Extract the message content
    message_content = formatted_message[message_start:message_end]
    
    print(message_content)
    return message_content

# Imports the needed functions for the practice mode
def practice_imports():
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
    from mode_settings import display_settings

# Imports the needed functions for the quiz mode
def quiz_imports():
    import cv2
    import mediapipe as mp
    import os
    import numpy as np
    import time
    import sys
    import random
    from hand_gesture_recognizer import recognize_letter
    from test_classifier import open_model
    from mode_settings import save_letter_quiz, save_word_quiz
    from mode_settings import display_settings, present_user_options_for_marks, letter_quiz_settings, word_quiz_settings
    from practice_mode import get_letter_image

def practice_feedback() -> str:
    practice_imports()
    from backend.gesture_model.old_practice_mode import get_directory, select_letter, initialize_camera, capture_and_process_frame, make_prediction, get_letter_image, update_and_display, practice_loop
    from backend.gesture_model.old_practice_mode import main as practice_main

    prompt = "The user has just finished a practice session of the fingerspelling alphabet"

    practice_mark, practice_history = practice_main()

    reponse = ask_gpt(prompt,practice_mark, practice_history)

def quiz_feedback():
    quiz_imports()
    from quiz_mode import main as quiz_main
    from quiz_mode import get_directory, initialize_camera as initialize_quiz_camera, capture_and_process_frame, make_prediction, update_and_display as update_and_display_letter_quiz, update_and_display_word, select_quiz_letter, select_quiz_word, quiz_words, type_of_quiz 

    prompt = "The user has just finished a quiz session of the fingerspelling alphabet, use the marks type to determine if it was an alphabet quiz or word quiz."
    quiz_mark = quiz_main()

    reponse = ask_gpt(prompt,quiz_mark, None)

def kind_gpt(username, task, context):
    gpt_role = f"""You are a kind and friendly American Sign Language Finger Spelling teacher. You want to give supportive feedback based on marks an errors, or just enouragement.
                You are called to respond at different stages in the program, be friendly and kind. 

                Here is some context about the program: {common_prompts()[0]}
    """
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system",
        "content": f"{gpt_role}"
        },

        {"role": "user",
        "content": f"""The username is {username}, your task is:{task}. Here is your extra context {context} """
        } ]
    )

    raw_message = str(completion.choices[0].message)
    formatted_message = raw_message.replace('\\n', '\n')
    
    # Split the string to extract the message part
    message_start = formatted_message.find('content="') + len('content="')
    message_end = formatted_message.find('", role=')

    # Extract the message content
    message_content = formatted_message[message_start:message_end]
    
    return message_content
 
def common_prompts():
    give_program_explanation = """ Give the user a kind and friendly explanation and summer of what SignSpell AI is. Speak as a friendly AI who knows it was created by Hamidat Bello and relies on GPT. 
                                    Here is information about SignSpell AI written by Hamidat Bello (UAlberta CS Student in her second year):

                                    SignSpell AI: Interactive Educational Tool for Learning Sign Language
                                    
                                    About the Project:
                                    Hello! I'm currently developing an interactive web-based platform that's designed to make learning sign language both engaging and effective. My project leverages machine learning for real-time gesture recognition and integrates the OpenAI GPT-4 API for smart, adaptive feedback. The goal is to create a dynamic environment where users can learn and practice sign language through interactive video feedback.

                                    Features to be Implemented
                                    Real-Time Gesture Recognition: I'm using computer vision technologies to interpret sign language gestures.
                                    AI-Powered Feedback: Utilizing OpenAI's GPT-4 API to provide personalized feedback and learning tips.
                                    Interactive Lessons: Working on creating engaging modules for practicing sign language.
                                    Progress Tracking: Planning to implement features to track and adapt to individual learning progress.
                                    Technology Stack
                                    Frontend: I'm considering HTML, CSS, JavaScript, and potentially a framework like React or Angular.
                                    Backend: The backend is being developed in Python, using Flask. I'm using OpenCV gesture recognition.
                                    APIs: Integrating with OpenAI's GPT-4 API for an enhanced learning experience.
                                    Database: I am using SQL for managing user data.

                                    The github link is: https://github.com/hamidatb/SignSpell-AI
                                    They can reach out to collaborate here with Hamidat (SWE, backend, project management): https://www.linkedin.com/in/hamidatbello/ (Im friendly and open to collaborate on projects or talk about tech)
                                    They can reach out here to collaborate with Zusi (frontend, UI/UX): https://www.linkedin.com/in/zusiarebun/

                                    """
    
    common_responses = [give_program_explanation]
    return common_responses

def introduction_loop():
    
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

    username = input("\nHi! :) Welcome to SignSpell AI! What's your name?: ").title().strip()
    introduction_options = "Ask what they would like to do? 1 = Run program, 2 = Have the info about SignSpell AI shown to them, 3 = Quit, 4 = Hear a random positive fact"
    print(f"\n{kind_gpt(username, introduction_options, None)}\n")
    introduction_options = int(input("Enter your response here (int):"))
    if introduction_options == 1:
        while True:
            run_type_prompt = f"Ask What type of session {username} like to do today (at this time)? Options: Practice Mode, Letter quiz, Word quiz, Wuit. (input p,l, w, or q). Give the user a structured list"
            print(f"\n{kind_gpt(username, run_type_prompt, None)}\n")

            run_type = input("Please enter your choice: ").strip().lower()
            if run_type == "p":
                practice_feedback()
                break
            elif run_type in ("l", "w"):
                quiz_feedback()
                break
            elif run_type == "q":
                nice_bye = "The user decided to quit SignSpell early, give them a friendly goodbye."
                print(f"\n{kind_gpt(username, nice_bye, None)}\n")
                break
            else:
                print("Invalid session type entered. Please input either (p, l, w or q).")
    elif introduction_options == 2:
        program_info_prompt = "Explain SignSpell AI in a friendly manner to the user"
        program_information = common_prompts()[0]
        print(f"\n{kind_gpt(username, program_info_prompt,program_information)}\n")
    elif introduction_options == 3:
        nice_bye_prompt = "The user decided to quit SignSpell early, give them a friendly goodbye."
        print(f"\n{kind_gpt(username, nice_bye_prompt, None)}\n")
    elif introduction_options == 4:
        fun_fact_prompt = "Give a friendly random positive fact about ASl and technology. Cite your source and be personable."
        print(f"\n{kind_gpt(username,fun_fact_prompt, None)}\n")
    else:
        sys.exit("Invalid option chosen. Bye! :)")

def main():
    introduction_loop()

if __name__ == "__main__":
    main()