import os
import numpy as np
import extract_feats.opensmile as of
import extract_feats.librosa as lf
import models
import utils

def predict(config, audio_path: str, model) -> None:
    """
    预测音频情感

    Args:
        config: 配置项
        audio_path (str): 要预测的音频路径
        model: 加载的模型
    """

    #utils.play_audio(audio_path)
    print(audio_path)
    def find_max_indices(arr):
        sorted_arr = sorted(((val, idx) for idx, val in enumerate(arr)), reverse=True)
        max_index = sorted_arr[0][1]
        second_max_index = sorted_arr[1][1]
        return max_index, second_max_index


    if config.feature_method == 'o':
        # 一个玄学 bug 的暂时性解决方案
        of.get_data(config, audio_path, train=False)
        test_feature = of.load_feature(config, train=False)
    elif config.feature_method == 'l':
        test_feature = lf.get_data(config, audio_path, train=False)

    result = model.predict(test_feature)
    result_prob = model.predict_proba(test_feature)
    max_index, second_max_index = find_max_indices(result_prob)
    difference = abs(result_prob[max_index]-result_prob[second_max_index])
    if (result_prob[max_index] >= 0.5 or difference > 0.15):
        print('Recogntion: ', config.class_labels[int(result)])
    else:
        print('Recogntion: ', config.class_labels[max_index], config.class_labels[second_max_index]) 
    print('Probability: ', result_prob)
    utils.radar(result_prob, config.class_labels)
    


if __name__ == '__main__':
    folder_path = 'datasets/CN/angry'
    wav_files = [file for file in os.listdir(folder_path) if file.endswith('.wav')]
    for wav_file in wav_files:
        audio_path = os.path.join(folder_path, wav_file)
        config = utils.parse_opt()
        model = models.load(config)
        predict(config, audio_path, model)
