from PyQt5 import QtWidgets
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush
from PyQt5.QtCore import Qt

# My files
from src.utils import get_screen_dimensions
from src.ui.base_gui import BaseGUI

screen_width, screen_height = get_screen_dimensions()


class PredictorGUI(BaseGUI):
    def __init__(self, controller):
        super().__init__(controller)
        self.prediction = None
        self.coordinates = {
            0: (0, 0, screen_width/2, screen_height/2),
            1: (screen_width/2, 0, screen_width, screen_height/2),
            2: (0, screen_height/2, screen_width/2, screen_height),
            3: (screen_width/2, screen_height/2, screen_width, screen_height),
        }

    def create_window(self):
        self.setWindowTitle('Predictor')
        self.resize(screen_width, screen_height)

        painter = QPainter(self)
        painter.drawRect(100, 100, 200, 200)

    def update_prediction(self, prediction):
        self.prediction = prediction
        # redraw the widget
        self.update()

    def paintEvent(self, event):
        if self.prediction is None:
            return
        painter = QPainter(self)
        painter.setPen(QPen(Qt.black, 5, Qt.SolidLine))
        painter.setBrush(QBrush(Qt.green, Qt.DiagCrossPattern))
        painter.drawRect(*(self.coordinates[self.prediction]))
