import cv2
import mediapipe as mp
import sys
import os
import pickle
import numpy as np

def get_directory():
    # Getting the directory where this script file is located
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    # Setting the directory where the data is stored, relative to this script file
    DATA_DIR = os.path.join(SCRIPT_DIR, "model.p")
    
    return SCRIPT_DIR, DATA_DIR

# Opening the pickle file
def open_model(SCRIPT_DIR, DATA_DIR):
    model_dict = pickle.load(open(DATA_DIR, "rb"))
    model = model_dict['model']
    return  model

def capture(model):
    # Note: The random classfiier model I built here is only expecting 42 features as input, so this only works with one hand so far
    # Capturing from my laptops main camera
    cap = cv2.VideoCapture(0)

    # This is the handtracking model from media pipe. For more details refer to create_dataset.py
    mp_hands = mp.solutions.hands 
    mp_drawings = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    labels_dict = {
        0:"A",
        1:"B",
        2:"L"}
    
    while True:
        data_loc = []
        x_ = []
        y_ = []

        ret, frame = cap.read()

        height, width, _  = frame.shape

        if not ret:
            sys.exit("Failed to grab frame")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Using the hand model from media pipe to analyze the handmarks of the hands in the fram if there are any
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawings.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS, # The predefined connections between hand landmarks (i.e., how the landmarks are linked to each other).
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
            
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_loc.extend([x, y])
                    x_.append(x)
                    y_.append(y)
            
            # Getting the corners of the rectangle containing the hand to keep things boxed in
            x1, y1 = int(min(x_) * width)-10, int(min(y_) * height)-10
            x2, y2 = int(max(x_) * width)-10, int(max(y_) * height)-10

            prediction = model.predict([np.asarray(data_loc)])
            predicted_charcter = labels_dict[int(prediction[0])]
        
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,0), 4)
            cv2.putText(frame, predicted_charcter, (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(frame, 'Press Q to quit!', (100, 50), cv2.FONT_HERSHEY_PLAIN, 2, (193, 182, 255), 3, cv2.LINE_AA)

        cv2.imshow("frame", frame)
        # Waiting 25 ms between each capture
        key = cv2.waitKey(1)

        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break  # Exit the function or use 'break' if in a loop


def main():
    SCRIPT_DIR, DATA_DIR = get_directory()
    model = open_model(SCRIPT_DIR, DATA_DIR)
    capture(model)

if __name__ == "__main__":
    main()