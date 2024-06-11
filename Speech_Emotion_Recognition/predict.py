import os
import numpy as np
import extract_feats.librosa as lf
import models
import utils

def predict(config, audio_path: str, model) -> None:
    """
    Predict the emotions of the audio

    Args:
        config
        audio_path (str)
        model: LSTM SVM MLP
    """

    # utils.play_audio(audio_path)
    def find_max_indices(arr):
        sorted_arr = sorted(((val, idx) for idx, val in enumerate(arr)), reverse=True)
        max_index = sorted_arr[0][1]
        second_max_index = sorted_arr[1][1]
        return max_index, second_max_index

    test_feature = lf.get_data(config, audio_path, train=False)

    result = model.predict(test_feature)
    result_prob = model.predict_proba(test_feature)
    print(audio_path)
    # 6-category
    '''
    max_index, second_max_index = find_max_indices(result_prob)
    difference = abs(result_prob[max_index]-result_prob[second_max_index])
    if (result_prob[max_index] >= 0.5 or difference > 0.15):
        print('Recogntion: ', config.class_labels[int(result)])
    else:
        print('Recogntion: ', config.class_labels[max_index], config.class_labels[second_max_index]) 
    print('Probability: ', result_prob)
    utils.radar(result_prob, config.class_labels)'''
    
    # 3-category
    # If you want to use 6-category prediction, you should comment out the following lines
    classlabel = ["positive", "negative", "neutral"]
    print('Recogntion: ', classlabel[int(result)])
    utils.radar(result_prob, classlabel)


if __name__ == '__main__':
    # Single file
    """audio_path = '/Users/zou/Renovamen/Developing/Speech-Emotion-Recognition/datasets/CASIA/angry/201-angry-liuchanhg.wav'

    config = utils.parse_opt()
    model = models.load(config)
    predict(config, audio_path, model)"""

    # Multiple files
    folder_path = 'datasets/IN_test'
    wav_files = [file for file in os.listdir(folder_path) if file.endswith('.wav')]
    for wav_file in wav_files:
        audio_path = os.path.join(folder_path, wav_file)
        config = utils.parse_opt()
        model = models.load(config)
        predict(config, audio_path, model)
