import numpy as np
from config import train_data_path
import os
import joblib


def train_model():
    from keras.models import Sequential
    from keras.layers import Dense, ReLU, Dropout, InputLayer
    print('Loading train data...')
    X, y = get_data()
    X = np.array(X)
    print('Training neural network...')
    model = Sequential([
        InputLayer(input_shape=(len(X[0]), )),
        Dense(100),
        Dropout(0.25),
        ReLU(),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(X, y, epochs=500, verbose=1)
    print('Training done')
    return model

    # model = MLPClassifier(hidden_layer_sizes=(
    #     64, 32), random_state=1, verbose=True, max_iter=30, learning_rate='constant', learning_rate_init=0.001)
    # predictions = model.fit(X, Y).predict(X)
    # accuracy = sum([1 if x[0] == y[0] and x[1] == y[1] else 0 for x,
    #                 y in zip(Y, predictions)])/len(Y)
    # print('Training done!')
    # return model, accuracy


def get_data():
    path = os.path.join(os.getcwd(), train_data_path, 'train_data.pkl')
    data = joblib.load(path)
    X = [(d[0][0] + d[0][1]).flatten() for d in data]
    y = [d[1] for d in data]
    return X, y


if __name__ == '__main__':
    model = train_model()
