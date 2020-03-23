import numpy as np
import os
import joblib
from threading import Lock
import json
import math
import time

# My files
import config as Config

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


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


last_model_number_lock = Lock()


def train_model(train_parameters):
    train_cnn_with_faces()
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


def train_cnn_with_faces():
    ...


def get_data(filename='train_data.pkl'):
    path = os.path.join(os.getcwd(), Config.train_data_path, filename)
    data = joblib.load(path)

    X = data[0]

    # only if i'm using the eyes
    # X = [(x[0].flatten(), x[1].flatten()) for x in X]
    # X = [np.concatenate(x) for x in X]

    y = data[1]
    return X, y


# TODO check if these parameters should be None
# TODO make sure that I'm saving models with score on validation data (not on train data)
def save_model(model, train_parameters=None, score=None, training_time=None):
    print(model)
    print('Saving model...')
    model_name = get_next_model_name()
    file_path = os.path.join(
        os.getcwd(), Config.models_directory_path, model_name)

    try:
        save_model_configuration(
            file_path, train_parameters, score, training_time)
    except Exception as e:
        print(f'Failed to save model as {model_name}: {e}')
    else:
        joblib.dump(model, file_path)
        print(f'Model saved as {model_name}')


def save_model_configuration(file_path, train_parameters, score, training_time):
    # save the configuration in the same folder with the same name, but in a json file
    file_path = file_path.split('.pkl')[0] + '.json'
    config = {
        "train_parameters": train_parameters,
        "score": score,
        "training_time": training_time
    }
    try:
        with open(file_path, 'w') as f:
            json.dump(config, f)
    except Exception as e:
        print(f'Could not save model configuration: {e}')
        raise


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
        "epochs": 5
    }
    res = train_model(train_parameters)
    save_model(res["model"], train_parameters={},
               score=res["fit_history"].history['loss'], training_time=res["training_time"])
