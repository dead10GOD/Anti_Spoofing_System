from flask import Flask, render_template, Response, redirect, url_for
import cv2
import torch
from model import DualChannelNet, predict_image, device

app = Flask(__name__)
camera = cv2.VideoCapture(0)

# Load the trained model
model_path = 'models/dual_channel_model.pth'
model = DualChannelNet()
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture')
def capture():
    success, frame = camera.read()
    if success:
        image_path = 'static/captured_image.jpg'
        cv2.imwrite(image_path, frame)
        confidence, prediction = predict_image(image_path, model, device)
        return render_template('result.html', prediction=prediction, confidence=confidence, image_path=image_path)
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
