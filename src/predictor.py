import numpy as np
import time
from threading import Thread

# My files
from src.utils import get_screen_dimensions
from src.data_processing import process_images
from src.trainer import get_best_trained_model
from src.ui.predictor_gui import PredictorGUI
from config import EYE_WIDTH, EYE_HEIGHT


class Predictor():
    def __init__(self, webcam_capturer):
        super().__init__()
        self.webcam_capturer = webcam_capturer
        self.gui = PredictorGUI(self)

    def ui_was_closed(self):
        ...

    def start(self):
        self.gui.show()
        Thread(target=self.predict).start()

    def predict(self):
        screen_size = get_screen_dimensions()

        print('Loading best trained model...')
        model = get_best_trained_model()
        if model is None:
            print('No trained models')
            self.gui.close()
            return

        while self.gui.isVisible():
            success, image = self.webcam_capturer.get_webcam_image()
            if success is False:
                print('Failed capturing image')
                continue

            X = process_images([image])
            # X = [(x[0].flatten(), x[1].flatten()) for x in X]
            # X = [np.concatenate(x) for x in X]
            # X = np.array(X)
            if len(X) == 0:
                continue
            X = [X[0][0].reshape(EYE_WIDTH, EYE_HEIGHT, 1)]
            X = np.array(X)
            print (X.shape)
            prediction = model.predict(X)[0]
            # For some reason, they're reversed
            prediction = 3 - prediction
            prediction = prediction.argmin()
            self.gui.update_prediction(prediction)
