import torch
import os
from flask import Flask, request, abort, jsonify, send_from_directory
from PIL import Image
from torchvision import transforms

from models import Net

app = Flask(__name__, static_url_path='/')

net = Net()
checkpoint = torch.load(
    './saved_models/model-2018-12-02T08-16-33-final.pt', map_location='cpu')
net.load_state_dict(checkpoint['model'])
net.eval()

data_transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize(250),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0], [255]),
], )


@app.route('/api/predict', methods=['POST'])
def predict():
    if not request.files or 'file' not in request.files:
        abort(400)
    f = request.files['file']
    img = Image.open(f)
    img = data_transform(img)
    # our model expect batch input, so we need to unsqueeze here
    img = torch.unsqueeze(img, 0)
    output = net(img)
    output = output * 50 + 100
    output = output.view(-1, 2)
    response = {
        "keypoints": output.tolist(),
    }
    return jsonify(response)


@app.route('/')
def index():
    root_dir = os.path.dirname(os.path.realpath(__file__))
    return send_from_directory(root_dir, 'index.html')


if __name__ == '__main__':
    app.run()
