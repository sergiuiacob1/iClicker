import tkinter as tk
import threading
from cv2 import cv2
import os
import logging


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


# def apply_function_per_thread(input, func, f_args=()):
#     """This functions applies the `func` function to every element from the `input` list and splits the work between threads.

#     This function is only suitable if the amount of work to be done is large!

#     The number of threads used is equal to the CPU's processors count"""
#     print(
#         f'Applying function {func} to input. Splitting work amongst {os.cpu_count()} threads.')
#     threads = []
#     start = 0
#     diff = len(input) // os.cpu_count()
#     for i in range(0, os.cpu_count()):
#         # the last thread gets whatever remains
#         if i == os.cpu_count() - 1:
#             end = len(input)
#         else:
#             end = start + diff
#         t = threading.Thread(target=thread_work, args=(
#             input, start, end, func, f_args))
#         threads.append(t)
#         start += diff

#     # start all threads
#     for t in threads:
#         t.start()
#     # wait all threads
#     for t in threads:
#         t.join()

# def thread_work(input, start, end, func, f_args):
#     input[start:end] = [func(x, *f_args) for x in input[start:end]]


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    os.makedirs('./logs', exist_ok=True)
    formatter = logging.Formatter('%(asctime)s: %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


if __name__ == '__main__':
    import time
    import math
    from multiprocessing import Pool
    def f(x): return math.sqrt(x) * 2 // 3
    count = 50000000

    a = [x for x in range(0, count + 1)]
    ss = time.time()
    a = [f(x)for x in a]
    print(f'Simple: {time.time() - ss}')

    # a = [x for x in range(0, 50000000 + 1)]
    # ss = time.time()
    # apply_function_per_thread(a, f, f_args=(3, 1))
    # print(f'threaded: {time.time() - ss}')
    a = [x for x in range(0, count + 1)]
    ss = time.time()
    with Pool(os.cpu_count()) as p:
        p.map(f, a)
    print(f'threaded: {time.time() - ss}')
