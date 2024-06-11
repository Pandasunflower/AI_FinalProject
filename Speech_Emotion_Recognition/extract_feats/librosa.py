import os
import re
import sys
import librosa
from random import shuffle
import numpy as np
from typing import Tuple, Union
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import utils

def features(X, sample_rate: float) -> np.ndarray:
    stft = np.abs(librosa.stft(X))

    # fmin => the minimum fundamental frequency of human speech
    # fmax => the maximum fundamental frequency of human speech
    pitches, magnitudes = librosa.piptrack(y=X, sr=sample_rate, S=stft, fmin=70, fmax=400)
    pitch = []
    for i in range(magnitudes.shape[1]):  # every time slot
        index = magnitudes[:, 1].argmax()
        pitch.append(pitches[index, i])

    # Estimate the tuning offset
    pitch_tuning_offset = librosa.pitch_tuning(pitches)
    pitchmean = np.mean(pitch)         # mean
    pitchstd = np.std(pitch)           # standard deviation
    pitchmax = np.max(pitch)           # maximum  
    pitchmin = np.min(pitch)           # minimum

    # Compute & normalize the spectral centroid 
    cent = librosa.feature.spectral_centroid(y=X, sr=sample_rate)
    cent = cent / np.sum(cent)
    meancent = np.mean(cent)           # mean
    stdcent = np.std(cent)             # standard deviation
    maxcent = np.max(cent)             # maximum

    # Compute the spectral flatness
    flatness = np.mean(librosa.feature.spectral_flatness(y=X))

    # Compute the MFCCs
    mfccs_ = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T
    mfccs = np.mean(mfccs_, axis=0)    # mean
    mfccsstd = np.std(mfccs_, axis=0)  # standard deviation
    mfccmax = np.max(mfccs_, axis=0)   # maximum

    # Compute the mean of the chroma features
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)  # stft => Short-Time Fourier Transform

    # Compute the mean of the Mel spectrogram features
    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)

    # Compute the mean of the spectral constrast features
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)

    # Compute the mean of the zero-crossing rate
    zerocr = np.mean(librosa.feature.zero_crossing_rate(X))

    # Compute the magnitude and phase of the STFT
    S, phase = librosa.magphase(stft)
    meanMagnitude = np.mean(S)         # mean
    stdMagnitude = np.std(S)           # standard deviation
    maxMagnitude = np.max(S)           # maximum

    # Compute the root mean square energy
    rmse = librosa.feature.rms(S=S)[0]
    meanrms = np.mean(rmse)            # mean
    stdrms = np.std(rmse)              # standard deviation
    maxrms = np.max(rmse)              # maximum 

    # all features
    ext_features = np.array([
        flatness, zerocr, meanMagnitude, maxMagnitude, meancent, stdcent,
        maxcent, stdMagnitude, pitchmean, pitchmax, pitchstd,
        pitch_tuning_offset, meanrms, maxrms, stdrms
    ])

    ext_features = np.concatenate((ext_features, mfccs, mfccsstd, mfccmax, chroma, mel, contrast))

    return ext_features

def extract_features(file: str, pad: bool = False) -> np.ndarray:
    X, sample_rate = librosa.load(file, sr=None)  # X => the audio signal data
    max_ = X.shape[0] / sample_rate
    # padding
    if pad:
        length = (max_ * sample_rate) - X.shape[0]
        X = np.pad(X, (0, int(length)), 'constant')
    return features(X, sample_rate)

def get_max_min(files: list) -> Tuple[float]:
    min_, max_ = 100, 0

    for file in files:
        sound_file, samplerate = librosa.load(file, sr=None)
        t = sound_file.shape[0] / samplerate   # time
        if t < min_:
            min_ = t
        if t > max_:
            max_ = t

    return max_, min_

def get_data_path(data_path: str, class_labels: list) -> list:
    """
    get all data path

    Args:
        data_path (str)
        class_labels (list)
    Returns:
        wav_file_path (list)
    """
    wav_file_path = []

    cur_dir = os.getcwd()
    sys.stderr.write('Curdir: %s\n' % cur_dir)
    os.chdir(data_path)

    # all folders
    for _, directory in enumerate(class_labels):
        os.chdir(directory)

        # .wav files under a folder
        for filename in os.listdir('.'):
            if not filename.endswith('wav'):
                continue
            filepath = os.path.join(os.getcwd(), filename)
            wav_file_path.append(filepath)

        os.chdir('..')
    os.chdir(cur_dir)

    shuffle(wav_file_path) # rearrange randomly  
    return wav_file_path

def load_feature(config, train: bool) -> Union[Tuple[np.ndarray], np.ndarray]:
    """
    Get features from "{config.feature_folder}/*.p" 

    Args:
        config
        train (bool)

    Returns:
        - X (Tuple[np.ndarray]): the label of training & testing features
        - X (np.ndarray): predict feature
    """
    feature_path = os.path.join(config.feature_folder, "train.p" if train == True else "predict.p")

    features = pd.DataFrame(
        data = joblib.load(feature_path),
        columns = ['file_name', 'features', 'emotion']
    )

    X = list(features['features'])
    Y = list(features['emotion'])

    # the path of the normalized model
    scaler_path = os.path.join(config.checkpoint_path, 'LSTM_CN_3_SCALER_LIBROSA.m') # change

    if train == True:
        # Normalize data
        scaler = StandardScaler().fit(X)
        # Save the normalized model
        utils.mkdirs(config.checkpoint_path)
        joblib.dump(scaler, scaler_path)
        X = scaler.transform(X)

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        return x_train, x_test, y_train, y_test

    else:
        # Load the normalized model
        scaler = joblib.load(scaler_path)
        X = scaler.transform(X)
        return X

def get_data(config, data_path: str, train: bool) -> Union[Tuple[np.ndarray], np.ndarray]:
    """
    Extract features of every audio
    
    Args:
        config
        data_path (str)
        train (bool)

    Returns:
        - train = True: 
            Train & test features
        - train = False: 
            Predict the feature
    """
    if train == True:
        # get all files      
        files = get_data_path(data_path, config.class_labels)
        max_, min_ = get_max_min(files)
        
        mfcc_data = []
        classlabels = ["positive", "negative", "neutral"]
        for file in files:
            # change 
            label = re.findall(r".*\\CN\\(.*)\\.*", file)[0]

            # 3-category
            
            if(label == "sad" or label == "neutral"):
                label = "neutral"
            elif(label == "angry" or label == "fear"):
                label = "negative"
            elif(label == "happy" or label == "surprise"):
                label = "positive"

            # Extract features
            features = extract_features(file, max_)
            #mfcc_data.append([file, features, config.class_labels.index(label)])
            mfcc_data.append([file, features, classlabels.index(label)])

    else:
        # Extract features
        features = extract_features(data_path)
        mfcc_data = [[data_path, features, -1]]

    # Save features
    feature_path = os.path.join(config.feature_folder, "train.p" if train == True else "predict.p")
    pickle.dump(mfcc_data, open(feature_path, 'wb'))

    return load_feature(config, train=train)
