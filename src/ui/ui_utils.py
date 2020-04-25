from PyQt5 import QtWidgets, QtGui
from cv2 import cv2

__doc__ = "Functions useful for PyQt GUI"


def build_button(name, tooltip, function, f_args = tuple()):
    """
    Builds and returns a QtWidgets.QPushButton with the given `name`, `tooltip` and `function`
    """
    button = QtWidgets.QPushButton(name)
    button.setToolTip(tooltip)
    button.clicked.connect(lambda _: function(f_args))
    return button


def get_qimage_from_cv2(cv2_image):
    """Receives an image that was captured using OpenCV.

    Returns a `QImage`
    """
    rgb_array = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_array.shape
    bytesPerLine = ch * w
    return QtGui.QImage(rgb_array.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
