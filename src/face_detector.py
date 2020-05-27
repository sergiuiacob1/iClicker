import dlib
from imutils import face_utils
import numpy as np

# my files
import config as Config
import src.utils as Utils

face_detector = None
face_predictor = None
stuff_was_initialized = False


def initialize_stuff():
    global stuff_was_initialized, face_detector, face_predictor
    stuff_was_initialized = True
    face_detector = dlib.get_frontal_face_detector()
    face_predictor = dlib.shape_predictor(Config.face_landmarks_path)


def extract_eyes(cv2_image):
    """Returns a list of images that contain the eyes extracted from the original image.

    First result is the left eye, second result is the right eye."""
    global stuff_was_initialized, face_detector, face_predictor
    if stuff_was_initialized == False:
        initialize_stuff()

    gray_image = Utils.convert_to_gray_image(cv2_image)
    rects = face_detector(gray_image, 0)
    if len(rects) > 0:
        shape = face_predictor(gray_image, rects[0])
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
    global stuff_was_initialized, face_detector
    if stuff_was_initialized == False:
        initialize_stuff()

    gray_image = Utils.convert_to_gray_image(cv2_image)
    rects = face_detector(gray_image, 0)
    if len(rects) > 0:
        # only for the first face found
        (x, y, w, h) = face_utils.rect_to_bb(rects[0])
        return cv2_image[y:y+h, x:x+w]
    return None


def extract_eye_strip(cv2_image):
    """Returns a horizontal image containing the two eyes extracted from the image"""
    global stuff_was_initialized, face_detector, face_predictor
    if stuff_was_initialized == False:
        initialize_stuff()

    gray_image = Utils.convert_to_gray_image(cv2_image)
    rects = face_detector(gray_image, 0)
    if len(rects) > 0:
        # only for the first face found
        shape = face_predictor(gray_image, rects[0])
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
    global stuff_was_initialized, face_detector, face_predictor
    if stuff_was_initialized == False:
        initialize_stuff()

    gray_image = Utils.convert_to_gray_image(cv2_image)
    rects = face_detector(gray_image, 0)
    if len(rects) > 0:
        shape = face_predictor(gray_image, rects[0])
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