import numpy as np
import time
from threading import Thread
from pynput.mouse import Controller

# My files
from src.utils import get_screen_dimensions, resize_cv2_image
from src.data_processing import extract_faces, extract_eye_strips, extract_thresholded_eyes
from src.trainer import get_best_trained_model
from src.ui.predictor_gui import PredictorGUI
import src.webcam_capturer as WebcamCapturer
from src.face_detector import is_mouth_opened
import config as Config

screen_width, screen_height = get_screen_dimensions()
_mouse_controller = Controller()
dx_value = screen_width * 0.01
dy_value = screen_height * 0.01


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
        print('Loading last trained model...')
        prediction_type = 'regression'
        trained_with = 'keras'
        data_used = f'eye_strips_regression.pkl'
        # data_used = f'eye_strips_{Config.grid_size}.pkl'
        # data_used = f'thresholded_eyes_{Config.grid_size}.pkl'
        # data_used = f'extracted_faces_{Config.grid_size}.pkl'
        model = get_best_trained_model(
            prediction_type=prediction_type, trained_with=trained_with, data_used=data_used, get_last_or_best="last")
        if model is None:
            self.gui.close()
            return

        # just to draw the initial cells
        self.gui.update_prediction(None)

        # async, move the mouse if it's necessary
        Thread(target=self._check_if_should_move_mouse).start()

        while self.gui.isVisible():
            time.sleep(1/Config.PREDICT_LOOK_FPS)
            success, image = WebcamCapturer.get_webcam_image()
            if success is False:
                print('Failed capturing image')
                continue

            try:
                if prediction_type == 'regression':
                    prediction = self.predict_regression(model, image)
                else:
                    if data_used.startswith('eye_strips'):
                        prediction = self.predict_based_on_eye_strips(
                            model, image)
                    elif data_used.startswith('extracted_faces'):
                        prediction = self.predict_based_on_extracted_face(
                            model, image)
                    elif data_used.startswith('thresholded_eyes'):
                        prediction = self.predict_based_on_thresholded_eyes(
                            model, image)
                    else:
                        prediction = None
                if prediction is None:
                    self.last_prediction = None
                    continue
            except Exception as e:
                print(f'Exception: {str(e)}')
            self.gui.update_prediction(prediction)
            self.last_prediction = prediction

    def _check_if_should_move_mouse(self):
        """Moves the mouse cursor if it's the case"""
        while self.gui.isVisible():
            time.sleep(1/Config.CHECK_MOUSE_MOVE_FPS)
            success, image = WebcamCapturer.get_webcam_image()
            if success is False:
                continue
            is_opened, ratio = is_mouth_opened(image)
            self.gui.update_mouth(is_opened)
            if is_opened == True:
                self._move_mouse()

    def _move_mouse(self):
        if self.last_prediction is None:
            return
        # 0 1 2 3 4 5 6 7 8
        dx = [-1, 0, 1, -1, 0, 1, -1, 0, 1]
        dy = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
        pos = _mouse_controller.position
        pos = (pos[0] + dx[self.last_prediction] * dx_value,
               pos[1] + dy[self.last_prediction] * dy_value)
        _mouse_controller.position = pos

    def predict_regression(self, model, img):
        # build data
        X = extract_eye_strips([img])
        if X[0] is None:
            return None
        input_shape = (Config.EYE_STRIP_HEIGHT, Config.EYE_STRIP_WIDTH, 1)
        X = list(map(lambda x: x.reshape(*input_shape), X))
        X = np.array(X)

        # predict
        prediction = model.predict(X)[0]
        # first value corresponds to width, second one to height]
        # TODO I should also treat the case in which the model was trained on a different resolution

        # find the corresponding cell
        # make sure I don't exit the screen with the prediction
        prediction[0] = max(prediction[0], 0)
        prediction[1] = max(prediction[1], 0)
        prediction[0] = min(prediction[0], screen_width - 1)
        prediction[1] = min(prediction[1], screen_height - 1)
        dx = screen_width / Config.grid_size
        dy = screen_height / Config.grid_size
        prediction = int(prediction[1] // dy *
                         Config.grid_size + prediction[0] // dx)
        return prediction

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
