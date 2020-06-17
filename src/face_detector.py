import dlib
from imutils import face_utils
import numpy as np
from threading import Lock

# my files
import config as Config
import src.utils as Utils

_face_detector = None
_face_predictor = None
_initialised = False
_initialize_lock = Lock()


def _initialize_detectors():
    global _initialised, _face_detector, _face_predictor
    _initialize_lock.acquire()
    _face_detector = dlib.get_frontal_face_detector()
    _face_predictor = dlib.shape_predictor(Config.face_landmarks_path)
    _initialised = True
    _initialize_lock.release()


def _detectors_are_initialised():
    _initialize_lock.acquire()
    res = (_initialised == True)
    _initialize_lock.release()
    return res


def extract_eyes(cv2_image):
    """Returns a list of images that contain the eyes extracted from the original image.

    First result is the left eye, second result is the right eye."""
    global _face_detector, _face_predictor
    if _detectors_are_initialised() == False:
        _initialize_detectors()

    gray_image = Utils.convert_to_gray_image(cv2_image)
    rects = _face_detector(gray_image, 0)
    if len(rects) > 0:
        shape = _face_predictor(gray_image, rects[0])
        shape = face_utils.shape_to_np(shape)

        eyes = []
        for eye in ["left_eye", "right_eye"]:
            # get the points for the contour
            (eye_start, eye_end) = face_utils.FACIAL_LANDMARKS_IDXS[eye]
            contour = shape[eye_start:eye_end]
            # get the upper left point, lower right point for this eye
            start = [min(contour, key=lambda x: x[0])[0],
                     min(contour, key=lambda x: x[1])[1]]
            end = [max(contour, key=lambda x: x[0])[0],
                   max(contour, key=lambda x: x[1])[1]]
            # extract the current eye
            eyes.append(cv2_image[start[1]:end[1], start[0]:end[0]])
        return eyes

    return None


def extract_face(cv2_image):
    """Returns the face part extracted from the image"""
    global _face_detector
    if _detectors_are_initialised() == False:
        _initialize_detectors()

    gray_image = Utils.convert_to_gray_image(cv2_image)
    rects = _face_detector(gray_image, 0)
    if len(rects) > 0:
        # only for the first face found
        (x, y, w, h) = face_utils.rect_to_bb(rects[0])
        return cv2_image[y:y+h, x:x+w]
    return None


def extract_eye_strip(cv2_image):
    """Returns a horizontal image containing the two eyes extracted from the image"""
    global _face_detector, _face_predictor
    if _detectors_are_initialised() == False:
        _initialize_detectors()

    gray_image = Utils.convert_to_gray_image(cv2_image)
    rects = _face_detector(gray_image, 0)
    if len(rects) > 0:
        # only for the first face found
        shape = _face_predictor(gray_image, rects[0])
        shape = face_utils.shape_to_np(shape)
        (left_eye_start,
         left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (right_eye_start,
            right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        # get the contour
        start, end = min(left_eye_start, right_eye_start), max(
            left_eye_end, right_eye_end)
        strip = shape[start:end]
        # get the upper left point, lower right point
        start = [min(strip, key=lambda x: x[0])[0],
                 min(strip, key=lambda x: x[1])[1]]
        end = [max(strip, key=lambda x: x[0])[0],
               max(strip, key=lambda x: x[1])[1]]
        # go a little outside the bounding box, to capture more details
        distance = (end[0] - start[0], end[1] - start[1])
        # 20 percent more details on the X axis, 60% more details on the Y axis
        percents = [20, 60]
        for i in range(0, 2):
            start[i] -= int(percents[i]/100 * distance[i])
            end[i] += int(percents[i]/100 * distance[i])
        return cv2_image[start[1]:end[1], start[0]:end[0]]
    return None


def extract_eyes_for_heatmap(cv2_image):
    global _face_detector, _face_predictor
    if _detectors_are_initialised() == False:
        _initialize_detectors()

    gray_image = Utils.convert_to_gray_image(cv2_image)
    rects = _face_detector(gray_image, 0)
    if len(rects) > 0:
        shape = _face_predictor(gray_image, rects[0])
        shape = face_utils.shape_to_np(shape)

        eyes = []
        for eye in ["left_eye", "right_eye"]:
            # get the points for the contour
            (eye_start, eye_end) = face_utils.FACIAL_LANDMARKS_IDXS[eye]
            # increase a little bit the size of the eye
            contour = shape[eye_start:eye_end]
            # get the upper left point, lower right point for this eye
            start = [min(contour, key=lambda x: x[0])[0],
                     min(contour, key=lambda x: x[1])[1]]
            end = [max(contour, key=lambda x: x[0])[0],
                   max(contour, key=lambda x: x[1])[1]]
            # increase a little bit the size of the eye
            distance = (end[0] - start[0], end[1] - start[1])
            percents = [40, 40]
            for i in range(0, 2):
                start[i] -= int(percents[i]/100 * distance[i])
                end[i] += int(percents[i]/100 * distance[i])
            # extract the current eye
            eyes.append(cv2_image[start[1]:end[1], start[0]:end[0]])
        return eyes

    return None


def _extract_eye_strip_from_image(image, shape):
    """Returns the eye strip by already knowing the shape (facial landmarks) of the face"""
    (left_eye_start,
     left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (right_eye_start,
        right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    # get the contour
    start, end = min(left_eye_start, right_eye_start), max(
        left_eye_end, right_eye_end)
    strip = shape[start:end]
    # get the upper left point, lower right point
    start = [min(strip, key=lambda x: x[0])[0],
             min(strip, key=lambda x: x[1])[1]]
    end = [max(strip, key=lambda x: x[0])[0],
           max(strip, key=lambda x: x[1])[1]]
    # go a little outside the bounding box, to capture more details
    distance = (end[0] - start[0], end[1] - start[1])
    # 20 percent more details on the X axis, 60% more details on the Y axis
    percents = [20, 60]
    for i in range(0, 2):
        start[i] -= int(percents[i]/100 * distance[i])
        end[i] += int(percents[i]/100 * distance[i])
    return image[start[1]:end[1], start[0]:end[0]]


def _dist2d(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _is_mouth_opened(shape):
    """Uses the MAR (Mouth Aspect Ratio)"""
    cd = _dist2d(shape[61], shape[67])
    ef = _dist2d(shape[62], shape[66])
    gh = _dist2d(shape[63], shape[65])
    ab = _dist2d(shape[60], shape[64])
    mar = (cd + ef + gh) / (3 * ab)
    return bool(mar > Config.MAR_THRESHOLD), mar
    # up1 = _dist2d(shape[50], shape[61])
    # up2 = _dist2d(shape[51], shape[62])
    # up3 = _dist2d(shape[52], shape[63])
    # bottom1 = _dist2d(shape[67], shape[58])
    # bottom2 = _dist2d(shape[66], shape[57])
    # bottom3 = _dist2d(shape[65], shape[56])
    # m1 = _dist2d(shape[61], shape[67])
    # m2 = _dist2d(shape[62], shape[66])
    # m3 = _dist2d(shape[63], shape[65])
    # up_height = (up1 + up2 + up3) / 3
    # bottom_height = (bottom1 + bottom2 + bottom3) / 3
    # mouth_height = (m1 + m2 + m3) / 3
    # return bool(mouth_height > up_height + bottom_height), mouth_height / (up_height + bottom_height)


def _are_eyes_opened(shape):
    """Idea from here: https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/"""
    left_v1 = _dist2d(shape[43], shape[47])
    left_v2 = _dist2d(shape[44], shape[46])
    left_h = _dist2d(shape[42], shape[45])
    right_v1 = _dist2d(shape[37], shape[41])
    right_v2 = _dist2d(shape[38], shape[40])
    right_h = _dist2d(shape[36], shape[39])

    left_ratio = (left_v1 + left_v2) / (2 * left_h)
    right_ratio = (right_v1 + right_v2) / (2 * right_h)
    return (bool(left_ratio > Config.EAR_THRESHOLD), bool(right_ratio > Config.EAR_THRESHOLD))


def get_img_info(cv2_image):
    """Returns a dictionary with necessary info about the image:
    eye strip, if mouth is opened, if eyes are opened."""
    global _face_detector, _face_predictor
    res = {
        "image": cv2_image,
        "mouth_is_opened": [None, None],
        "eyes_are_opened": [None, None],
    }
    if _detectors_are_initialised() == False:
        _initialize_detectors()

    gray_image = Utils.convert_to_gray_image(cv2_image)
    rects = _face_detector(gray_image, 0)
    if len(rects) > 0:
        shape = _face_predictor(gray_image, rects[0])
        shape = face_utils.shape_to_np(shape)
        res["mouth_is_opened"] = _is_mouth_opened(shape)
        res["eyes_are_opened"] = _are_eyes_opened(shape)

    return res
