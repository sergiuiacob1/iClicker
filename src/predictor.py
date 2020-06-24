import json
import os
import numpy as np
import time
from threading import Thread
from pynput.mouse import Button, Controller

# My files
from src.utils import get_screen_dimensions, resize_cv2_image
from src.data_processing import extract_faces, extract_eye_strips, extract_thresholded_eyes
from src.trainer import get_best_trained_model
from src.ui.predictor_gui import PredictorGUI
import src.webcam_capturer as WebcamCapturer
from src.face_detector import get_img_info
import config as Config

_screen_width, _screen_height = get_screen_dimensions()
_mouse_controller = Controller()
_dx_value = _screen_width * Config.CURSOR_DX
_dy_value = _screen_height * Config.CURSOR_DY
_frames_closed = [0, 0]
_can_click = [True, True]
_prediction_info = None
_time_last_opened = [time.time(), time.time()]
_time_last_closed = [time.time(), time.time()]


def _clear_prediction_info():
    global _prediction_info
    _prediction_info = {
        "image": None,
        "mouth_is_opened": (None, 0),
        "eyes_are_opened": ((None, None), (0, 0)),
    }


class Predictor():
    def __init__(self):
        self.gui = PredictorGUI(self)
        _clear_prediction_info()
        self.prediction_type = 'regression'
        self.trained_with = 'keras'
        self.data_used = f'eye_strips_regression.pkl'
        # self.data_used = f'eye_strips_{Config.grid_size}.pkl'
        # self.data_used = f'thresholded_eyes_{Config.grid_size}.pkl'
        # self.data_used = f'extracted_faces_{Config.grid_size}.pkl'

    def ui_was_closed(self):
        self._gui_not_closed = False
        WebcamCapturer.stop_capturing()

    def start(self):
        self._gui_not_closed = True
        self.gui.show()
        WebcamCapturer.start_capturing()
        Thread(target=self.predict).start()

    def can_predict(self):
        path = os.path.join(os.getcwd(), Config.models_directory_path)
        if os.path.exists(path) is False:
            return False

        # Check if there is a model corresponding to the type of prediction I'm trying to
        models_info = [x for x in os.listdir(path) if x.endswith(".json")]
        if len(models_info) == 0:
            return False

        for x in models_info:
            model_info_path = os.path.join(path, x)
            with open(model_info_path, 'r') as f:
                data = json.load(f)

            if self.trained_with is not None:
                if data['trained_with'] != self.trained_with:
                    continue
            if self.data_used is not None:
                if data['data_used'] != self.data_used:
                    continue
            if data['grid_size'] != Config.grid_size:
                continue
            if data['prediction_type'] != self.prediction_type:
                continue

            # I fould at least 1 model that matches the description
            return True
        return False

    def predict(self):
        print('Loading last trained model...')
        prediction_type = self.prediction_type
        trained_with = self.trained_with
        data_used = self.data_used
        model = get_best_trained_model(
            prediction_type=prediction_type, trained_with=trained_with, data_used=data_used, get_last_or_best="last")
        if model is None:
            self.gui.close()
            return

        self.last_prediction = None
        Thread(target=self._update_info_for_prediction).start()
        Thread(target=self._check_info).start()
        Thread(target=self._update_prediction, args=(
            model, prediction_type, data_used)).start()

    def _update_prediction(self, model, prediction_type, data_used):
        """Currently uses eye strips to predict where the user is looking"""
        global _prediction_info
        # how much time an iteration should take
        interval = 1/Config.PREDICT_LOOK_FPS
        while self._gui_not_closed:
            start = time.time()
            image = _prediction_info["image"]
            if image is None:
                continue
            prediction = None
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
            except Exception as e:
                print(f'Exception: {str(e)}')
            self.gui.update_prediction(prediction)
            self.last_prediction = prediction
            end = time.time()
            if interval - (end - start) > 0:
                time.sleep(interval - (end - start))

    def _update_info_for_prediction(self):
        """Update the info necessary to make predictions.
        Info updated are things like eye strips, if mouth is opened, if eyes are opened etc."""
        global _prediction_info

        # this is how much an iteration should take
        interval = 1/Config.UPDATE_FPS
        while self.gui.isVisible():
            start = time.time()
            success, image = WebcamCapturer.get_webcam_image()
            if success is False:
                _clear_prediction_info()
                continue
            _prediction_info = get_img_info(image)
            end = time.time()
            if interval - (end - start) > 0:
                time.sleep(interval - (end - start))

    def _check_info(self):
        """Moves the mouse cursor if it's the case"""
        global _frames_closed, _prediction_info, _can_click, _time_last_opened, _time_last_closed

        # how much an iteration should take
        interval = 1/Config.UPDATE_FPS
        while self.gui.isVisible():
            start = time.time()
            self.gui.update_info(dict((k, _prediction_info[k]) for k in (
                'mouth_is_opened', 'eyes_are_opened')))
            if _prediction_info["mouth_is_opened"][0] == True:
                self._move_mouse()
            for i in range(2):
                if _prediction_info["eyes_are_opened"][0][i] == True:
                    _time_last_opened[i] = time.time()
                else:
                    _time_last_closed[i] = time.time()

            self._check_mouse_clicks()
            end = time.time()
            if interval - (end - start) > 0:
                time.sleep(interval - (end - start))

    def _check_mouse_clicks(self):
        global _can_click, _time_last_opened, _time_last_closed, _mouse_controller
        buttons = [Button.left, Button.right]
        for i in range(2):
            # if the eye i wants to click and if it's "more closed" than the other
            if time.time() - _time_last_opened[i] >= Config.EYE_CLICK_TIME and _can_click[i] == True \
                    and _prediction_info["eyes_are_opened"][1][i] < _prediction_info["eyes_are_opened"][1][1 - i]:
                # print(f'Am dat click cu {i}')
                _mouse_controller.press(buttons[i])
                _mouse_controller.release(buttons[i])
                _can_click[i] = False

            if time.time() - _time_last_closed[i] >= Config.EYE_CLICK_TIME * 2 and _can_click[i] == False:
                _can_click[i] = True
                # print(f'{i} can click!')

    def _move_mouse(self):
        global _mouse_controller
        if self.last_prediction is None:
            return
        # 0 1 2 3 4 5 6 7 8
        dx = [-1, 0, 1, -1, 0, 1, -1, 0, 1]
        dy = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
        pos = _mouse_controller.position
        pos = (pos[0] + dx[self.last_prediction] * _dx_value,
               pos[1] + dy[self.last_prediction] * _dy_value)
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
        prediction[0] = min(prediction[0], _screen_width - 1)
        prediction[1] = min(prediction[1], _screen_height - 1)
        dx = _screen_width / Config.grid_size
        dy = _screen_height / Config.grid_size
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
