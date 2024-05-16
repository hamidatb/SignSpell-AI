from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import os
from gesture_model.mode_settings import load_progress, save_progress
from gesture_model.practice_mode import main as practice_main
from gesture_model/quiz_mode import main as quiz_main

app = Flask(__name__)

# Ensure the camera is initialized only once
cap = cv2.VideoCapture(0)

@app.route('/')
def index():
    """Render the main page with options to start quiz or practice mode."""
    return render_template('index.html')

@app.route('/start_practice')
def start_practice():
    """Route to start practice mode."""
    user_progress = load_progress("user_progress.pkl")
    return practice_main(user_progress)

@app.route('/start_quiz')
def start_quiz():
    """Route to start quiz mode."""
    return quiz_main()

def gen_frames():
    """Generate frame by frame from camera."""
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/video_feed')
def video_feed():
    """Route to stream video from the camera."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Shut down the Flask application."""
    shutdown_server()
    return 'Server shutting down...'

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

if __name__ == '__main__':
    app.run(debug=True)
