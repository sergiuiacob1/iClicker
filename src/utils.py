import tkinter as tk
import threading
from cv2 import cv2
import os


def get_screen_dimensions():
    root = tk.Tk()
    root.withdraw()  # don't display the root window
    return root.winfo_screenwidth(), root.winfo_screenheight()


def run_function_on_thread(function, f_args=tuple()):
    threading.Thread(target=function, args=f_args).start()


def convert_to_gray_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


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


def get_binary_thresholded_image(cv2_image):
    img = convert_to_gray_image(cv2_image)
    img = cv2.medianBlur(img, 5)
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return img
