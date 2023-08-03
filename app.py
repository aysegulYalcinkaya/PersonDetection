from flask import Flask, render_template, request, redirect, url_for
import os
import os.path as osp
import cv2
from PIL import Image
import torch
import pandas as pd
import requests
import ultralytics

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/how-it-works')
def how_it_works():
    return render_template('how-it-works.html')

@app.route('/get-started')
def get_started():
    return render_template('get-started.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def results_parser(results):
    s = ""
    if results.pred[0].shape[0]:
        for c in results.pred[0][:, -1].unique():
            n = (results.pred[0][:, -1] == c).sum()  # detections per class
            s += f"{n} {results.names[int(c)]}{'s' * (n > 1)}, "  # add to string
    return s


def process_video(model, video):
    video_name = osp.basename(video)
    outputpath = osp.join('data/video_output', video_name)

    # Create A Dir to save Video Frames
    os.makedirs('data/video_frames', exist_ok=True)
    frames_dir = osp.join('data/video_frames', video_name)
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video)
    frame_count = 0

    while True:
        frame_count += 1
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = model(frame)
        print(results_parser(result))
        result.render()
        image = Image.fromarray(result.ims[0])

        image.save(osp.join(frames_dir, f'{frame_count}.jpg'))
    cap.release()
    # convert frames in dir to a single video file without using ffmeg
    image_folder = frames_dir
    video_name = outputpath

    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('get_started'))

    file = request.files['file']

    if file.filename == '':
        return redirect(url_for('get_started'))

    if file and allowed_file(file.filename):
        # Save the uploaded file to the 'uploads' folder
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

        # Process the video using the ML model
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        CFG_MODEL_PATH = "models/yolov5s.pt"
        model = torch.hub.load('ultralytics/yolov5', 'custom',
                               path=CFG_MODEL_PATH, force_reload=True, device='cpu')
        frame_count = process_video(model,video_path)

        return f"Video uploaded and processed. Total frames: {frame_count}"

    else:
        return "Invalid file format. Please upload a valid video file."


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run()
