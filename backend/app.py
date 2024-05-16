# backend/app.py
from flask import Flask, render_template, Response
from socketio_setup import socketio
import cv2
import base64
import threading
import time
from gesture_model.quiz_mode import main as quiz_main
from gesture_model.quiz_mode import handle_quiz_answer

app = Flask(__name__)
socketio.init_app(app)

cap = cv2.VideoCapture(0)

@app.route('/')
def index():
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
            _, buffer = cv2.imencode('.jpg', frame)
            frame_encoded = base64.b64encode(buffer).decode('utf-8')
            socketio.emit('video_frame', {'frame': frame_encoded})
        socketio.sleep(0.1)

@socketio.on('connect')
def on_connect():
    socketio.emit('response', {'message': 'Connected to server!'})

@socketio.on('start_quiz')
def start_quiz():
    print("Quiz started")
    socketio.emit('response', {'message': 'Starting quiz...'})
    threading.Thread(target=quiz_main).start()
    
@socketio.on('quiz_answer')
def quiz_answer(data):
    handle_quiz_answer(data)

if __name__ == '__main__':
    socketio.start_background_task(target=send_frame_to_socketio)
    socketio.run(app, debug=True)
    cv2.destroyAllWindows()
