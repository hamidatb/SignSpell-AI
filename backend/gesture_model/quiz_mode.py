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
from mode_settings import display_settings, quiz_words
from practice_mode import get_directory, initialize_camera, get_letter_image, capture_and_process_frame, make_prediction, update_and_display, select_letter


def quiz_letters(model, progress, file_path, settings, images_dir):
    pass


def quiz_words(model, progress, file_path, settings, images_dir):
    cap, hands = initialize_camera()
    total_attempts = 0
    total_correct = 0
    letter_accuracies = {chr(65 + i): {'attempts': 0, 'correct': 0} for i in range(26)}

    
    for i in range(settings["Amount of letters to practice"]):
        pass




def main():
    SCRIPT_DIR, MODEL_DIR, IMAGES_DIR = get_directory()
    model = open_model(SCRIPT_DIR, MODEL_DIR)
    settings = practice_settings()
    progress_file = "user_progress.pkl"
    user_progress = load_progress(progress_file)
    quiz_loop(model, user_progress, progress_file, settings, IMAGES_DIR)

if __name__ == "__main__":
    main()