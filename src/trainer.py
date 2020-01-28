import numpy as np
import os
import joblib
from threading import Lock

# My files
import src.config as Config


def get_last_model_number():
    path = os.path.join(os.getcwd(), Config.models_directory_path)
    os.makedirs(path, exist_ok=True)
    files = os.listdir(path)
    files = [x for x in files if x.endswith('.pkl')]
    if len(files) == 0:
        return 0
    print(files)
    numbers = [int(x.split('_')[1].split('.pkl')[0]) for x in files]
    if len(numbers) == 0:
        return 0
    return max(numbers)


last_model_number = get_last_model_number()
last_model_number_lock = Lock()


def train_model():
    # Lazy import so the app starts faster
    from keras.models import Sequential
    from keras.layers import Dense, ReLU, Dropout

    print('Loading train data...')
    X, y = get_data()
    X = np.array(X)

    print('Training neural network...')
    model = Sequential([
        Dense(100, input_shape=(len(X[0]),),
              kernel_initializer='glorot_uniform'),
        ReLU(),
        Dense(16, kernel_initializer='lecun_uniform', activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=2, verbose=1)
    print('Training done')
    return model


def get_data():
    path = os.path.join(os.getcwd(), Config.train_data_path, 'train_data.pkl')
    data = joblib.load(path)
    print(f'Loaded {len(data)} items')
    X = [(d[0][0] + d[0][1]).flatten() for d in data]
    y = [d[1] for d in data]
    return X, y


# TODO check if these parameters should be None
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
        "mse_score": score,
        "training_time": training_time
    }
    try:
        with open(file_path, 'w') as f:
            json.dump(config, f)
    except Exception as e:
        print(f'Could not save model configuration: {e}')
        raise


def get_next_model_name():
    global last_model_number, last_model_number_lock
    last_model_number_lock.acquire()

    last_model_number += 1
    model_name = f'model_{last_model_number}.pkl'

    last_model_number_lock.release()
    return model_name


def get_best_trained_model():
    # Lazy library loading so app starts faster
    from keras.models import load_model
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
        if data['score'][-1] < min_score:
            min_score = data['score'][-1]
            best_model_name = x.split('.json')[0] + '.pkl'

    try:
        # keras.backend.clear_session()
        # This is necessary if I want to load a model multiple times
        clear_session()
        model = joblib.load(os.path.join(path, best_model_name))
        print(f'Best model chosen: {best_model_name}')
    except Exception as e:
        print(f'Could not load model {best_model_name}: {str(e)}')
        return None

    return model


if __name__ == '__main__':
    model = train_model()

    model.save('./models/model.pkl')
    from keras.models import load_model
    same_model = load_model('./models/model.pkl')
