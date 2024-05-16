
from flask import Flask, render_template, Response, request, redirect, url_for
from flask_socketio import SocketIO, emit
import cv2
import os
import sqlite3
from gesture_model.mode_settings import load_progress, save_progress
from gesture_model.practice_mode import main as practice_main
from gesture_model.quiz_mode import main as quiz_main
from flask_session import Session
from werkzeug.security import check_password_hash, generate_password_hash
import base64
import io
from PIL import Image

app = Flask(__name__)
socketio = SocketIO(app)
socketio_instance = socketio

# Ensure the camera is initialized only once
cap = cv2.VideoCapture(0)

@app.route('/')
def index():
    """Render the main page with options to start quiz or practice mode."""
    return render_template('quiz.html')

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

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def send_frame_to_socketio():
    while True:
        success, frame = cap.read()
        if success:
            # Encode the frame in JPEG format
            _, buffer = cv2.imencode('.jpg', frame)
            # Convert the frame to base64 to send it via Socket.IO
            frame_encoded = base64.b64encode(buffer).decode('utf-8')
            # Emit the frame to the 'video_frame' event
            socketio.emit('video_frame', {'frame': frame_encoded})
        socketio.sleep(0.1)

@socketio.on('connect')
def on_connect():
    emit('response', {'message': 'Connected to server!'})

if __name__ == '__main__':
    socketio.start_background_task(target=send_frame_to_socketio)
    socketio.run(app, debug=True)