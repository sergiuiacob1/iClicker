import numpy as np
from numpy.random import permutation
import os
import joblib
from threading import Lock
import json
import math
import time

# My files
import config as Config
from src.utils import setup_logger


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
last_model_number_lock = Lock()
train_logger = setup_logger('train_logger', 'train.log')


def get_last_model_number():
    path = os.path.join(os.getcwd(), Config.models_directory_path)
    os.makedirs(path, exist_ok=True)
    files = os.listdir(path)
    files = [x for x in files if x.endswith('.pkl')]
    if len(files) == 0:
        return 0
    numbers = [int(x.split('_')[1].split('.pkl')[0]) for x in files]
    if len(numbers) == 0:
        return 0
    return max(numbers)


def train_model(train_parameters):
    st = time.time()
    f = train_cnn_with_keras
    # f = train_mlp
    # Loading the data specific to the config's grid size
    # f_args = (f'eye_strips_{Config.grid_size}.pkl',
    #           (Config.EYE_STRIP_HEIGHT, Config.EYE_STRIP_WIDTH, 1))
    f_args = (f'extracted_faces_{Config.grid_size}.pkl',
              (Config.FACE_HEIGHT, Config.FACE_WIDTH, 1))
    # f_args = (f'thresholded_eyes_{Config.grid_size}.pkl',)
    print(f'Training model with {f} on {f_args[0]}')
    train_logger.info(f'Training model with {f} on {f_args[0]}')
    res = f(*f_args)
    res['training_time'] = time.time() - st
    train_logger.info(f'Training with {f} took {time.time() - st} seconds')
    return res


def train_mlp(which_data):
    # Lazy import so the app starts faster
    from keras.models import Sequential
    from keras.layers import Dense, ReLU, Dropout, Flatten
    from keras.optimizers import RMSprop, Adam
    from keras import regularizers
    from keras import backend as K

    print(f'Loading train data: {which_data}')
    X, y = get_data(which_data)
    # shuffle the data
    perm = permutation(len(X))
    X = X[perm]
    y = y[perm]
    initial_input_shape = X[0].shape
    n = X[0].shape[0] * X[0].shape[1]
    X = np.array([x.flatten() for x in X])

    print('Training neural network...')
    model = Sequential([
        Dense(100, input_shape=(n,), kernel_initializer='glorot_uniform',
              kernel_regularizer=regularizers.l2(0.01)),
        Dropout(0.5),
        ReLU(),
        Dense(128, kernel_initializer='glorot_uniform',
              kernel_regularizer=regularizers.l2(0.01)),
        Dropout(0.5),
        ReLU(),
        Dense(64, kernel_initializer='glorot_uniform',
              kernel_regularizer=regularizers.l2(0.01)),
        ReLU(),
        Dense(Config.grid_size * Config.grid_size, activation='softmax')
    ])
    loss = 'categorical_crossentropy'
    opt = Adam()
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    fit_history = model.fit(
        X, y, epochs=train_parameters["epochs"], batch_size=train_parameters["batch_size"], verbose=1, validation_split=0.2)

    return {
        "model": model,
        "type": "MLP",
        "trained_with": "keras",
        "data_used": which_data,
        "input_shape": initial_input_shape,
        "dataset_size": len(X),
        "grid_size": Config.grid_size,
        f"train_loss_{loss}": fit_history.history['loss'],
        f"test_loss_{loss}": fit_history.history['val_loss'],
        "train_accuracy": fit_history.history['acc'],
        "test_accuracy": fit_history.history['val_acc'],
    }


def train_cnn_with_keras(which_data, input_shape):
    # Lazy import so the app starts faster
    from keras.models import Sequential
    from keras.layers import Dense, ReLU, Dropout, Conv2D, MaxPooling2D, Flatten
    from keras.optimizers import RMSprop, Adam, Adagrad, SGD
    from keras import backend as K

    print('Loading train data...')
    X, y = get_data(which_data)
    # shuffle the data
    n = len(X)
    perm = permutation(n)
    X = X[perm]
    y = y[perm]

    # the number of rows = the height of the image!
    X = list(map(lambda x: x.reshape(*input_shape), X))
    X = np.array(X)
    initial_input_shape = X[0].shape

    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3),
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(ReLU())
    model.add(Conv2D(32, kernel_size=(3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(ReLU())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(Config.grid_size * Config.grid_size, activation='softmax'))

    opt = Adam()
    loss = 'categorical_crossentropy'
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    start_time = time.time()
    fit_history = model.fit(
        X, y, epochs=train_parameters["epochs"], batch_size=train_parameters["batch_size"], validation_split=0.2, verbose=1)
    end_time = time.time()
    print('Training done')
    return {
        "model": model,
        "type": "CNN",
        "trained_with": "keras",
        "data_used": which_data,
        "input_shape": initial_input_shape,
        "dataset_size": len(X),
        "grid_size": Config.grid_size,
        f"train_loss_{loss}": fit_history.history['loss'],
        f"test_loss_{loss}": fit_history.history['val_loss'],
        "train_accuracy": fit_history.history['acc'],
        "test_accuracy": fit_history.history['val_acc'],
    }


def train_cnn(which_data):
    import torch.nn as nn
    import torch.optim as optim

    class ConvNet(nn.Module):
        def __init__(self, output_dim):
            super(ConvNet, self).__init__()

            self.conv = nn.Sequential()
            self.conv.add_module("conv_1", nn.Conv2d(1, 16, kernel_size=3))
            self.conv.add_module("maxpool_1", nn.MaxPool2d(kernel_size=2))
            self.conv.add_module("relu_1", nn.ReLU())
            self.conv.add_module("conv_2", nn.Conv2d(16, 32, kernel_size=3))
            self.conv.add_module("maxpool_2", nn.MaxPool2d(kernel_size=2))
            self.conv.add_module("relu_2", nn.ReLU())

            self.fc = nn.Sequential()
            self.fc.add_module("fc_1", nn.Linear(320, 128))
            self.fc.add_module("relu_3", nn.ReLU())
            self.fc.add_module("fc_2", nn.Linear(50, output_dim))
            self.fc.add_module("softmax_1", nn.Softmax())

        def forward(self, x):
            x = self.conv.forward(x)
            x = x.view(-1, 320)
            return self.fc.forward(x)

    cnn = ConvNet(4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters())

    X, y = get_data(which_data)
    n = len(X)
    perm = permutation(n)
    X = X[perm]
    y = y[perm]

    return {
        "model": cnn,
        "trained_with": "pytorch",
        "data_used": which_data,
        "score": None,
    }


def get_data(filename='train_data.pkl'):
    path = os.path.join(os.getcwd(), Config.train_data_path, filename)
    data = joblib.load(path)

    X = data[0]

    # only if i'm using the eyes
    # X = [(x[0].flatten(), x[1].flatten()) for x in X]
    # X = [np.concatenate(x) for x in X]

    y = data[1]
    return X, y


# TODO make sure that I'm saving models with score on validation data (not on train data)... or both
def save_model(model, train_parameters={}):
    """
    `model` is a dictionary containing information about the model and the model itself.
    """
    print(model["model"])
    print('Saving model...')
    model_name = get_next_model_name()
    model_path = os.path.join(
        os.getcwd(), Config.models_directory_path, model_name)

    # save information about this model
    conf_path = model_path.split('.pkl')[0] + '.json'
    config = model.copy()
    config["train_parameters"] = train_parameters
    del config["model"]

    try:
        with open(conf_path, 'w') as f:
            json.dump(config, f)
    except Exception as e:
        print(f'Could not save model configuration: {e}')
        return
    # finally, save the serialized model
    if config["trained_with"] == "pytorch":
        import torch
        torch.save(model["model"].state_dict(), model_path)
    else:
        joblib.dump(model["model"], model_path)
    print(f'Model saved as {model_name}')


def get_next_model_name():
    global last_model_number_lock
    last_model_number_lock.acquire()

    last_model_number = get_last_model_number()
    last_model_number += 1
    model_name = f'model_{last_model_number}.pkl'

    last_model_number_lock.release()
    return model_name


def get_best_trained_model(trained_with=None, data_used=None, get_last_or_best="last"):
    print(
        f'Loading model trained with {trained_with} on {data_used} for a grid size of {Config.grid_size}')
    # Check that the directory with models exists
    if os.path.exists(Config.models_directory_path) is False:
        return None

    # Get the best model based on its score
    path = os.path.join(os.getcwd(), Config.models_directory_path)
    models_info = [x for x in os.listdir(path) if x.endswith(".json")]
    if len(models_info) == 0:
        return None

    min_score = math.inf
    best_model_name = ''
    for x in models_info:
        model_info_path = os.path.join(path, x)
        with open(model_info_path, 'r') as f:
            data = json.load(f)
        if data['test_accuracy'] is None:
            continue
        if trained_with is not None:
            if data['trained_with'] != trained_with:
                continue
        if data_used is not None:
            if data['data_used'] != data_used:
                continue
        if data['grid_size'] != Config.grid_size:
            continue
        should_update = False
        if get_last_or_best == "best":
            if data['test_accuracy'][-1] < min_score:
                should_update = True
        else:
            # extract the numbers from the model names and compare those to see which model is the latest
            temp = int(x.split('_')[1].split('.')[0])
            if best_model_name == '':
                best_no = 0
            else:
                best_no = int(best_model_name.split('_')[1].split('.')[0])
            if temp > best_no:
                should_update = True
        if should_update:
            min_score = data['test_accuracy'][-1]
            best_model_name = x.replace('.json', '.pkl')
            best_model_info = data

    if best_model_name == '':
        print('No model was found')
        return None
    try:
        if best_model_info['trained_with'] == 'keras':
            # Lazy library loading so app starts faster
            from keras.backend import clear_session
            # This is necessary if I want to load a model multiple times
            clear_session()
            model = joblib.load(os.path.join(path, best_model_name))
        else:
            # TODO maybe trained with pytorch, do this case too
            return None
    except Exception as e:
        print(f'Could not load model {best_model_name}: {str(e)}')
        return None

    print(f'Model chosen: {best_model_name}')
    return model


if __name__ == '__main__':
    train_parameters = {
        "epochs": 50,
        "batch_size": 32,
    }
    res = train_model(train_parameters)
    save_model(res, train_parameters=train_parameters)
