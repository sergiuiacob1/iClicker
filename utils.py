import tkinter as tk
import threading
from PyQt5 import QtWidgets
from cv2 import cv2
from PyQt5 import QtGui
import math
import functools


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


def run_function_on_thread(function, f_args=tuple()):
    threading.Thread(target=function, args=f_args).start()


def convert_to_gray_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def sort_points_counter_clockwise(points):
    if len(points) == 0:
        return []
    def point_comparator(a, b):
        a1 = (math.degrees(math.atan2(
            a[0] - x_center, a[1] - y_center)) + 360) % 360
        a2 = (math.degrees(math.atan2(
            b[0] - x_center, b[1] - y_center)) + 360) % 360
        return (int)(a1 - a2)

    x_center = sum([x[0] for x in points])/len(points)
    y_center = sum([x[1] for x in points])/len(points)

    return sorted(points, key=functools.cmp_to_key(point_comparator))


def resize_cv2_image(cv2_image, scale=None, fixed_dim=None):
    if scale is None and fixed_dim is None:
        return cv2_image
    if fixed_dim is not None:
        res = cv2.resize(cv2_image, fixed_dim, interpolation=cv2.INTER_AREA)
    else:
        height, width, _ = cv2_image.shape
        res = cv2.resize(cv2_image, (width*scale, height*scale),
                         interpolation=cv2.INTER_AREA)
    return res
