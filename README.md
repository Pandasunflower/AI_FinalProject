# Speech Emotion Recognition
### 1. How to build and run?
#### (1) cd into Speech_Emotion_Recognition
#### (2) You python version must be 3.8
#### (3) pip install -r requirements.txt
#### (4) python preprocess.py --config configs/example.yaml(replace the yaml you want)
#### (5) python train.py --config configs/example.yaml(replace the yaml you want)
#### (6) python predict.py --config configs/example.yaml(replace the yaml you want)
-----------------------------------------------------------------------------------------
### 2. File structures
#### (1) checkpoints: The model you have trained.
#### (2) configs: The yaml(model type) you can choose.
#### (3) datasets: The training data, including English, Franch, Deutsch, Chinese, Japan and Hindi.
#### (4) extra_feats: The method for extracting features.
#### (5) features: The features extracting from datasets.
#### (6) graphs: The result of training dataset accuracy and testing data.
#### (7) models: The implementation of LSTM, MLP and SVM.
#### (8) utils: Something help us to plot figures, read file, and so on.
#### (9) predict.py: The method for prediction.
#### (10) preprocess.py: The method for preprocessing data into the same form.
#### (11) train.py: The method for training our models.
#### (12) requirements.py: Libraries you need to install.



