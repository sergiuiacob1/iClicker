from PyQt5 import QtWidgets
from src.utils import get_screen_dimensions


class Predictor(QtWidgets.QMainWindow):
    def __init__(self, webcam):
        super().__init__()
        self.webcam = webcam
        self.create_window()

    def create_window(self):
        self.setWindowTitle('Predictor')
        self.webcam_image = QtWidgets.QLabel()
        self.setCentralWidget(self.webcam_image)
