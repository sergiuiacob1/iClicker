from PyQt5 import QtWidgets
from webcam_capturer import WebcamCapturer
from utils import get_screen_dimensions


class Predictor(QtWidgets.QWidget):
    def __init__(self, model):
        super().__init__()
        self.create_window()

    def create_window(self):
        self.resize(*get_screen_dimensions())
        # self.setLayout(QtWidgets.)
