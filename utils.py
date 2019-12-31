import tkinter as tk
from PyQt5 import QtWidgets
from cv2 import cv2
from PyQt5 import QtGui


def get_screen_dimensions():
    root = tk.Tk()
    root.withdraw()  # don't display the root window
    return root.winfo_screenwidth(), root.winfo_screenheight()


def build_button(name, tooltip, function):
    """
    Builds and retungs a QtWidgets.QPushButton with the given `name`, `tooltip` and `function`
    """
    button = QtWidgets.QPushButton(name)
    button.setToolTip(tooltip)
    button.clicked.connect(function)
    return button


def build_sample_image(cv2_image):
    """Receives an image that was captured using OpenCV.

    Returns a `QImage`
    """
    rgb_array = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_array.shape
    bytesPerLine = ch * w
    return QtGui.QImage(rgb_array.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
