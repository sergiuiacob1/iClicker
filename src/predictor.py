import numpy as np
import time
from threading import Thread

# My files
from src.utils import get_screen_dimensions, resize_cv2_image
from src.data_processing import process_images, extract_faces, extract_eye_strips
from src.trainer import get_best_trained_model
from src.ui.predictor_gui import PredictorGUI
import src.webcam_capturer as WebcamCapturer
import config as Config


class Predictor():
    def __init__(self):
        super().__init__()
        self.gui = PredictorGUI(self)

    def ui_was_closed(self):
        WebcamCapturer.stop_capturing()

    def start(self):
        self.gui.show()
        WebcamCapturer.start_capturing()
        Thread(target=self.predict).start()

    def predict(self):
        screen_size = get_screen_dimensions()

        print('Loading best trained model...')
        trained_with = 'keras'
        data_used = 'eye_strips.pkl'
        model = get_best_trained_model(
            trained_with=trained_with, data_used=data_used)
        if model is None:
            print('No trained models')
            self.gui.close()
            return

        while self.gui.isVisible():
            success, image = WebcamCapturer.get_webcam_image()
            if success is False:
                print('Failed capturing image')
                continue

            if data_used == 'eye_strips.pkl':
                prediction = self.predict_based_on_eye_strips(model, image)
            elif data_used == 'extracted_faces.pkl':
                prediction = self.predict_based_on_extracted_face(model, image)
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
        print(prediction, prediction.argmin())
        prediction = prediction.argmin()
        return 3 - prediction

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
        print(prediction, prediction.argmin())
        prediction = prediction.argmin()
        return 3 - prediction
