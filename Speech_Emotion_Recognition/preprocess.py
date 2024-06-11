import extract_feats.librosa as lf
from utils import parse_opt

"""Extract features of the audio in the dataset and Save them."""

if __name__ == '__main__':
    config = parse_opt()

    lf.get_data(config, config.data_path, train=True)