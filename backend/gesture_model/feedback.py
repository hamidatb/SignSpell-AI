import os
import sys
from openai import OpenAI
from .practice_mode import get_directory
from dotenv import load_dotenv
from socketio_setup import socketio  # Absolute import


# Loading the envirnoment variables from the .env file (API key)
# Note: Only I have access to my specific .env file. If you'd like to run this part, copythis repo to your device and add your own .env file.
load_dotenv()

try:
    api_key = os.getenv('SIGNSPELL_API_KEY')
except ValueError:
    raise ValueError("No OPENAI_API_KEY found in environment variables")

client = OpenAI(api_key=api_key)

# The main function interacting with the openAI API
def ask_gpt(task_needed, marks=None, mark_history=None):
    gpt_role = "You are a kind and friendly American Sign Language Finger Spelling teacher. You want to give supportive feedback based on marks an errors, or just encouragement. Users are trying to use the program to learn ASL, and you're the kind feedback"

    messages = [
        {"role": "system", "content": gpt_role}
    ]

    if marks and mark_history:
        user_message = f"{task_needed} these are their marks on this attempt: {marks}, these were their previous marks: {mark_history}"
    elif marks:
        user_message = f"{task_needed} these are their marks on this attempt: {marks}"
    else:
        user_message = task_needed

    messages.append({"role": "user", "content": user_message})

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    # Properly access the message content
    message_content = completion.choices[0].message.content  # Use attribute access instead of dictionary access

    print(f"\n\nThe message content: {message_content}\n\n")
    return message_content

def kind_gpt(task, context):
    gpt_role = f"""You are a kind and friendly American Sign Language Finger Spelling teacher. You want to give supportive feedback based on marks an errors, or just encouragement.
                You are called to respond at different stages in the program, be friendly and kind. 

                Here is some context about the program: {common_prompts()[0]}
    """
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"{gpt_role}"},
            {"role": "user", "content": f"Your task is:{task}. Here is your extra context {context}. Refer to the user as 'you'"}
        ]
    )

    message_content = completion.choices[0].message.content

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
                                    They can reach out to collaborate here with Hamidat (SWE, backend, UI/UX, project management): https://www.linkedin.com/in/hamidatbello/ (Im friendly and open to collaborate on projects or talk about tech)
                                    They can reach out here to collaborate with Zusi (frontend, UI/UX): https://www.linkedin.com/in/zusiarebun/

                                    """
    
    common_responses = [give_program_explanation]
    return common_responses

def introduction_loop():
    # This function will return an introductory message
    introduction_options = "Say 'Hello! Welcome to SignSpell AI.' Ask what they would like to do? 1 = Run program, 2 = Have the info about SignSpell AI shown to them, 3 = I don't know, 4 = Hear a random positive fact"
    return kind_gpt(introduction_options, None)


def continue_loop(last_message_to_user):
    # This function will return an introductory message
    continue_options = f"""
                Don't say hello. Be brief but nice. No greeting. The user has already sent you a message. This was your last response to them {last_message_to_user}. Now, ask what they'd
                like to do next.  1 = Run program, 2 = Have the info about SignSpell AI shown to them, 3 = I don't know, 4 = Hear a random positive fact, or they can exit the chat using the x button."
                """
    return kind_gpt(continue_options, None)


def handle_user_choice(choice):
    if choice == 1:
        run_type_prompt = f"Tell the user. lease use the navigation bar or homepage to select the type of run you'd like to do today. (Either ASL letter quiz or an ASL fingerspelling word quiz or a practice session on the ASL letters)"
        return kind_gpt(run_type_prompt, None)
    elif choice == 2:
        program_info_prompt = "Explain SignSpell AI in a friendly manner to the user"
        program_information = common_prompts()[0]
        return kind_gpt(program_info_prompt, program_information)
    elif choice == 3:
        other_prompt = "The user doesn't know what they want, acknowledge that, tell them to have a good day and add a random ASL statistic."
        return kind_gpt(other_prompt, None)
    elif choice == 4:
        fun_fact_prompt = "Give a friendly random positive fact about ASL and technology. Cite your source and be personable."
        return kind_gpt(fun_fact_prompt, None)
    else:
        return "Invalid option chosen. Bye! :)"


def start_chat():
    # This is for running in the terminal!
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    username = input("\nHi! :) Welcome to SignSpell AI! What's your name?: ").title().strip()
    intro_message = introduction_loop(username)
    print(f"\n{intro_message}\n")
    return username, intro_message

def main():
    username, intro_message = start_chat()
    while True:
        choice = int(input("Enter your response here (int): "))
        response_message = handle_user_choice(choice)
        print(f"\n{response_message}\n")
        if choice in [1, 2, 3, 4]:
            break

if __name__ == "__main__":
    main()