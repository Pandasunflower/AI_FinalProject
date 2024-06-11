from .dnn import LSTM
from .ml import SVM, MLP

def make(config, n_feats: int):
    """
    build the model

    Args:
        config: Configuration
        n_feats (int): feature amount
    """
    if config.model == 'svm':
        model = SVM.make(params=config.params)
    elif config.model == 'mlp':
        model = MLP.make(params=config.params)
    elif config.model == 'lstm':
        model = LSTM.make(
            input_shape = n_feats,
            rnn_size = config.rnn_size,
            hidden_size = config.hidden_size,
            dropout = config.dropout,
            n_classes = 3,# if you want to use 6-category prediction, this should be 6
            lr = config.lr
        )


    return model


_MODELS = {
    'lstm': LSTM,
    'mlp': MLP,
    'svm': SVM
}

def load(config):
    return _MODELS[config.model].load(
        path = config.checkpoint_path,
        name = config.checkpoint_name
    )
