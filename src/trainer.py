import numpy as np
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
    f = train_cnn_with_faces_keras
    res = f()
    res['training_time'] = time.time() - st
    train_logger.info(f'Training with {f} took {time.time() - st} seconds')
    return res
    # # Lazy import so the app starts faster
    # from keras.models import Sequential
    # from keras.layers import Dense, ReLU, Dropout, Conv2D, MaxPooling2D, Flatten
    # from keras.optimizers import RMSprop
    # from keras import backend as K

    # print('Loading train data...')
    # X, y = get_data()
    # n = len(X)
    # X = np.array(X)
    # y = np.array(y)

    # # the number of rows = the height of the image!
    # input_shape = (Config.WEBCAM_IMAGE_HEIGHT, Config.WEBCAM_IMAGE_WIDTH, 1)
    # X = list(map(lambda x: x.reshape(*input_shape), X))
    # X = np.array(X)

    # # model = Sequential([
    # #     Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
    # #            activation='relu', input_shape=input_shape, data_format='channels_last'),
    # #     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    # #     Conv2D(64, (5, 5), activation='relu'),
    # #     MaxPooling2D(pool_size=(2, 2)),
    # #     Flatten(),
    # #     Dense(64, kernel_initializer='glorot_uniform', activation='relu'),
    # #     Dense(4, activation='softmax')
    # # ])
    # model = Sequential()
    # model.add(Conv2D(16, kernel_size=(3, 3),
    #                  activation='relu',
    #                  input_shape=input_shape))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(32, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # # model.add(Dropout(0.5))
    # model.add(Dense(4, activation='softmax'))

    # model.compile(optimizer='adam',
    #               loss='categorical_crossentropy', metrics=['accuracy'])
    # start_time = time.time()
    # fit_history = model.fit(
    #     X, y, epochs=train_parameters["epochs"], batch_size=32, verbose=1)
    # end_time = time.time()
    # print('Training done')

    # # print('Loading train data...')
    # # X, y = get_data()
    # # X = np.array(X)
    # # y = np.array(y)

    # # print('Training neural network...')
    # # model = Sequential([
    # #     Dense(100, input_shape=(len(X[0]),),
    # #           kernel_initializer='glorot_uniform'),
    # #     Dropout(0.5),
    # #     ReLU(),
    # #     Dense(16, kernel_initializer='glorot_uniform'),
    # #     ReLU(),
    # #     Dense(4, activation='softmax')
    # # ])
    # # rmsprop = RMSprop(lr=0.001)
    # # model.compile(optimizer='adagrad',
    # #               loss='categorical_crossentropy', metrics=['accuracy'])
    # # start_time = time.time()
    # # fit_history = model.fit(X, y, epochs=train_parameters["epochs"], verbose=1)
    # # end_time = time.time()
    # # print('Training done')
    # return {
    #     "model": model,
    #     "fit_history": fit_history,
    #     "training_time": math.floor(end_time - start_time)
    # }


def train_cnn_with_faces_keras():
    # Lazy import so the app starts faster
    from keras.models import Sequential
    from keras.layers import Dense, ReLU, Dropout, Conv2D, MaxPooling2D, Flatten
    from keras.optimizers import RMSprop, Adam, Adagrad
    from keras import backend as K

    print('Loading train data...')
    X, y = get_data('extracted_faces.pkl')
    n = len(X)

    # the number of rows = the height of the image!
    input_shape = (Config.FACE_HEIGHT, Config.FACE_WIDTH, 1)
    X = list(map(lambda x: x.reshape(*input_shape), X))
    X = np.array(X)

    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3),
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(ReLU())
    model.add(Conv2D(32, kernel_size=(3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(ReLU())
    # model.add(Conv2D(64, kernel_size=(4, 4)))
    # model.add(MaxPooling2D(pool_size=(3, 3)))
    # model.add(ReLU())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))

    opt = Adagrad()
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    start_time = time.time()
    fit_history = model.fit(
        X, y, epochs=train_parameters["epochs"], batch_size=32, verbose=1)
    end_time = time.time()
    print('Training done')
    return {
        "model": model,
        "trained_with": "keras",
        "score": fit_history.history['loss']
    }


def train_cnn_with_faces():
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

    X, y = get_data('extracted_faces.pkl')
    n = len(X)
    print(n)

    return {
        "model": cnn,
        "trained_with": "pytorch",
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


def get_best_trained_model():
    # Lazy library loading so app starts faster
    from keras.backend import clear_session

    # Check that the directory with models exists
    if os.path.exists(Config.models_directory_path) is False:
        return None

    # Get the best model based on its score
    path = os.path.join(os.getcwd(), Config.models_directory_path)
    models_info = [x for x in os.listdir(path) if x.endswith(".json")]
    if len(models_info) == 0:
        return None

    min_score = math.inf
    for x in models_info:
        model_info_path = os.path.join(path, x)
        with open(model_info_path, 'r') as f:
            data = json.load(f)
        if data['score'] is None:
            continue
        if data['score'][-1] < min_score:
            min_score = data['score'][-1]
            best_model_name = x.split('.json')[0] + '.pkl'

    # TODO delete this!!!
    best_model_name = 'model_6.pkl'

    try:
        # This is necessary if I want to load a model multiple times
        clear_session()
        model = joblib.load(os.path.join(path, best_model_name))
        print(f'Best model chosen: {best_model_name}')
    except Exception as e:
        print(f'Could not load model {best_model_name}: {str(e)}')
        return None

    return model


if __name__ == '__main__':
    train_parameters = {
        "epochs": 250
    }
    res = train_model(train_parameters)
    # res["fit_history"].history['loss']
    save_model(res, train_parameters=train_parameters)
