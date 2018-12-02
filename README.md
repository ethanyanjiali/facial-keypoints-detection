## Set Up
1. Install dependencies `pip install -r requirements.txt`
2. Configure your editor to use flake8 as linter
3. Download dataset
    ```bash
    mkdir /data
    wget -P /data/ https://s3.amazonaws.com/video.udacity-data.com/topher/2018/May/5aea1b91_train-test-data/train-test-data.zip
    unzip -n /data/train-test-data.zip -d /data
    ```