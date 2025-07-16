import os
from flask import Flask, render_template, request, send_from_directory
import cv2
import torch

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    download_link = None
    if request.method == 'POST':
        video = request.files['video']
        if video and video.filename.endswith('.mp4'):
            input_path = os.path.join(UPLOAD_FOLDER, video.filename)
            output_path = os.path.join(OUTPUT_FOLDER, 'output.mp4')
            video.save(input_path)
            detect_persons(input_path, output_path)
            download_link = '/static/output.mp4'
    return render_template('index.html', download_link=download_link)

def detect_persons(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        person_only = results.pred[0][results.pred[0][:, 5] == 0]
        results.pred[0] = person_only
        frame = results.render()[0]
        out.write(frame)

    cap.release()
    out.release()

@app.route('/static/<filename>')
def serve_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
