# backend/app.py
from flask import Flask, render_template, Response, request, jsonify
from socketio_setup import socketio
import cv2
import base64
import threading
import time
from gesture_model.quiz_mode import quiz_main
from gesture_model.quiz_mode import handle_quiz_answer
from gesture_model.practice_mode import main as practice_main
from gesture_model.practice_mode import handle_practice_answer
from gesture_model.feedback import handle_user_choice, introduction_loop, continue_loop


quiz_running = False  # Shared variable to indicate quiz state
practice_running = False

app = Flask(__name__)
socketio.init_app(app)

cap = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/quiz')
def quiz():
    return render_template('quiz.html')

@app.route('/practice')
def practice():
    return render_template('practice.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def send_frame_to_socketio():
    while True:
        global quiz_running, practice_running
        success, frame = cap.read()
        if success:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_encoded = base64.b64encode(buffer).decode('utf-8')
            socketio.emit('video_frame', {'frame': frame_encoded})
        
        if quiz_running or practice_running:
            socketio.sleep(10)  # While playing the game, update the frame less frequently.
        else:
            socketio.sleep(0)  # Update the frame more frequently when not playing the game.

@socketio.on('connect')
def on_connect():
    socketio.emit('response', {'message': 'Connected to server!'})

@socketio.on('start_quiz')
def start_quiz():
    global quiz_running
    quiz_running = True  # Set quiz_running to True when the quiz starts
    print("\nQuiz started\n")
    socketio.emit('response', {'message': 'Starting quiz...'})
    threading.Thread(target=quiz_main).start()

@socketio.on('start_practice')
def start_practice():
    global practice_running
    practice_running = True  # Set practice_running to True when practice starts
    print("\nPractice started\n")
    socketio.emit('response', {'message': 'Starting practice...'})
    threading.Thread(target=practice_main).start()


@app.route('/terminate_quiz', methods=['POST'])
def terminate_quiz():
    global quiz_running
    quiz_running = False  # Set quiz_running to False to stop the quiz
    print("Quiz terminated.")
    return jsonify({"message": "Quiz terminated"})

@app.route('/terminate_practice', methods=['POST'])
def terminate_practice():
    global practice_running
    practice_running = False  # Set practice_running to False to stop the practice
    print("Practice terminated.")
    return jsonify({"message": "Practice terminated"})

@socketio.on('quiz_answer')
def quiz_answer(data):
    global quiz_running
    quiz_running = handle_quiz_answer(data)

@socketio.on('practice_answer')
# Handle the practice answer
def practice_answer(data):
    handle_practice_answer(data)


@socketio.on('start_chat')
def start_chat():
    # Stat the chat workflow
    intro_message = introduction_loop()
    print(intro_message)
    socketio.emit('chat_response', {'response': intro_message})

@socketio.on('chat_message')
def handle_chat_message(data):
    message = data['message']
    print(message)
    global username
    try:
        choice = int(message)
        response_message = handle_user_choice(choice)
    except ValueError:
        response_message = "Please enter a valid option (1, 2, 3, or 4)."
    socketio.emit('chat_response', {'response': response_message})
    
def continue_chat(response_message):
    cont_message = continue_loop(response_message)
    print(cont_message)
    socketio.emit('chat_response', {'response': cont_message})


if __name__ == '__main__':
    socketio.start_background_task(target=send_frame_to_socketio)
    socketio.run(app, debug=True)

    cv2.destroyAllWindows()
