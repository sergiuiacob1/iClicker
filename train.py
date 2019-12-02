import numpy as np
from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler


def trainModel(data):
    X, Y = getData(data)
    print('Training neural network...')
    model = MLPClassifier(hidden_layer_sizes=(
        32, 16, 4), random_state=1, verbose=True, max_iter=30, learning_rate='constant', learning_rate_init=0.01)
    predictions = model.fit(X, Y).predict(X)
    accuracy = sum([1 if x[0] == y[0] and x[1] == y[1] else 0 for x,
                    y in zip(Y, predictions)])/len(Y)
    # forest = RandomForestClassifier(
    #     n_estimators=100, random_state=1, verbose=1)
    # multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
    # predictions = multi_target_forest.fit(X, Y).predict(X)
    return model, accuracy


def getData(data):
    print('Getting X and Y...')
    X = np.array(list(map(lambda x: np.array(x.image).flatten(), data)))
    Y = np.array(list(map(lambda x: (x.horizontal, x.vertical), data)))
    # Normalizing X
    print('Normalizing data...')
    X = MinMaxScaler().fit_transform(X)
    return X, Y
