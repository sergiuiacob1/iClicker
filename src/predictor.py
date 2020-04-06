import numpy as np
import time
from threading import Thread

# My files
from src.utils import get_screen_dimensions, resize_cv2_image
from src.data_processing import extract_faces, extract_eye_strips, extract_thresholded_eyes
from src.trainer import get_best_trained_model
from src.ui.predictor_gui import PredictorGUI
import src.webcam_capturer as WebcamCapturer
import config as Config


class Predictor():
    def __init__(self):
        self.gui = PredictorGUI(self)

    def ui_was_closed(self):
        WebcamCapturer.stop_capturing()

    def start(self):
        self.gui.show()
        WebcamCapturer.start_capturing()
        Thread(target=self.predict).start()

    def predict(self):
        screen_size = get_screen_dimensions()

        print('Loading last trained model...')
        trained_with = 'keras'
        data_used = f'eye_strips_{Config.grid_size}.pkl'
        # data_used = f'thresholded_eyes_{Config.grid_size}.pkl'
        # data_used = f'extracted_faces_{Config.grid_size}.pkl'
        model = get_best_trained_model(
            trained_with=trained_with, data_used=data_used, get_last_or_best="last")
        if model is None:
            print('No trained models')
            self.gui.close()
            return

        # just to draw the initial cells
        self.gui.update_prediction(None)

        while self.gui.isVisible():
            success, image = WebcamCapturer.get_webcam_image()
            if success is False:
                print('Failed capturing image')
                continue

            if data_used.startswith('eye_strips'):
                prediction = self.predict_based_on_eye_strips(model, image)
            elif data_used.startswith('extracted_faces'):
                prediction = self.predict_based_on_extracted_face(model, image)
            elif data_used.startswith('thresholded_eyes'):
                prediction = self.predict_based_on_thresholded_eyes(
                    model, image)
            else:
                prediction = None
            if prediction is None:
                continue
            self.gui.update_prediction(prediction)

    def predict_based_on_extracted_face(self, model, img):
        # build data
        X = extract_faces([img])
        if X[0] is None:
            return None
        input_shape = (Config.FACE_HEIGHT, Config.FACE_WIDTH, 1)
        X = list(map(lambda x: x.reshape(*input_shape), X))
        X = np.array(X)

        # predict
        prediction = model.predict(X)[0]
        # predictions come in reversed
        prediction = prediction[::-1]
        prediction = prediction.argmin()
        return prediction

    def predict_based_on_eye_strips(self, model, img):
        # build data
        X = extract_eye_strips([img])
        if X[0] is None:
            return None
        input_shape = (Config.EYE_STRIP_HEIGHT, Config.EYE_STRIP_WIDTH, 1)
        X = list(map(lambda x: x.reshape(*input_shape), X))
        X = np.array(X)

        # predict
        prediction = model.predict(X)[0]
        # predictions come in reversed
        prediction = prediction[::-1]
        prediction = prediction.argmin()
        return prediction

    def predict_based_on_thresholded_eyes(self, model, img):
        # build data
        X = extract_thresholded_eyes([img])
        if X[0] is None:
            return None
        X = np.array([x.flatten() for x in X])

        # predict
        prediction = model.predict(X)[0]
        # predictions come in reversed
        prediction = prediction[::-1]
        prediction = prediction.argmin()
        return prediction
