from tensorflow.keras.utils import to_categorical
import extract_feats.librosa as lf
import models
from utils import parse_opt

def train(config) -> None:
    """
    Training model

    Args:
        config

    Returns:
        model: LSTM SVM MLP
    """

    # Get features done by preprocess.py
    x_train, x_test, y_train, y_test = lf.load_feature(config, train=True)        

    # Build the model
    model = models.make(config=config, n_feats=x_train.shape[1])

    # Train the model
    print('----- start training', config.model, '-----')
    if config.model in ['lstm']:
        y_train, y_val = to_categorical(y_train), to_categorical(y_test)  # One-Hot Encoding
        model.train(
            x_train, y_train,
            x_test, y_val,
            batch_size = config.batch_size,
            n_epochs = config.epochs
        )
    else:
        model.train(x_train, y_train)
    print('----- end training ', config.model, ' -----')

    # Evaluate model
    model.evaluate(x_test, y_test)
    
    # Save model
    model.save(config.checkpoint_path, config.checkpoint_name)


if __name__ == '__main__':
    config = parse_opt()
    train(config)
