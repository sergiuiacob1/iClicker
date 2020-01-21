from PyQt5 import QtWidgets, QtGui
import config as Config
import utils as Utils
from ui.ui_utils import get_qimage_from_cv2
from utils import get_binary_thresholded_image


class EyeWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Eye')
        self.setLayout(QtWidgets.QVBoxLayout())
        self.eye = QtWidgets.QLabel()
        self.layout().addWidget(self.eye)
        # TODO delete below
        self.resize(200, 200)

    def update(self, cv2_image, eye_contour):
        x_min = min([x[0] for x in eye_contour])
        x_max = max([x[0] for x in eye_contour])
        y_min = min([x[1] for x in eye_contour])
        y_max = max([x[1] for x in eye_contour])

        eye_portion = cv2_image[y_min:y_max, x_min:x_max]
        eye_portion = Utils.resize_cv2_image(
            eye_portion, fixed_dim=(Config.EYE_WIDTH, Config.EYE_HEIGHT))
        eye_portion = get_binary_thresholded_image(eye_portion)
        q_image = get_qimage_from_cv2(eye_portion)
        self.eye.setPixmap(QtGui.QPixmap.fromImage(q_image))
