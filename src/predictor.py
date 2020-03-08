import numpy as np
import time
from threading import Thread

# My files
from src.utils import get_screen_dimensions, resize_cv2_image
from src.data_processing import process_images
from src.trainer import get_best_trained_model
from src.ui.predictor_gui import PredictorGUI
import src.webcam_capturer as WebcamCapturer
from config import EYE_WIDTH, EYE_HEIGHT, WEBCAM_IMAGE_HEIGHT, WEBCAM_IMAGE_WIDTH


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
        model = get_best_trained_model()
        if model is None:
            print('No trained models')
            self.gui.close()
            return

        while self.gui.isVisible():
            success, image = WebcamCapturer.get_webcam_image()
            if success is False:
                print('Failed capturing image')
                continue

            image = resize_cv2_image(image, fixed_dim=(
                WEBCAM_IMAGE_WIDTH, WEBCAM_IMAGE_HEIGHT))
            X = process_images([image])
            input_shape = (WEBCAM_IMAGE_HEIGHT, WEBCAM_IMAGE_WIDTH, 1)
            X = list(map(lambda x: x.reshape(*input_shape), X))
            X = np.array(X)
            # X = [(x[0].flatten(), x[1].flatten()) for x in X]
            # X = [np.concatenate(x) for x in X]
            # X = np.array(X)
            if len(X) == 0:
                continue
            X = np.array(X)
            prediction = model.predict(X)[0]
            # For some reason, they're reversed
            prediction = 3 - prediction
            prediction = prediction.argmin()
            self.gui.update_prediction(prediction)
