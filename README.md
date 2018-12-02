## Set Up
1. Activate virtualenv, and install dependencies `pip install -r requirements.txt`
2. Configure your editor to use flake8 as linter
3. Download facial keypoints dataset from Udacity `make download`
4. Start training and save models: `python train.py`

## Usage
1. Update the model path in `serve.py`
1. Start local server: `python serve.py`
2. Visit `http://locahost:5000/`
3. Upload a face image from the web UI
![example](/example.png)