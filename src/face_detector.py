import dlib
from imutils import face_utils
import config as Config
import src.utils as Utils

# TODO make a singleton here with lazy loading and threaded responses
# TODO actually, don't use a singleton; keep it simple.


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(
                Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class FaceDetector(metaclass=Singleton):
    def __init__(self):
        # TODO test if this is truly a singleton
        self._face_detector = dlib.get_frontal_face_detector()
        self._face_predictor = dlib.shape_predictor(
            Config.face_landmarks_path)

    def get_eye_contours(self, cv2_image):
        """Returns a list of eye contours from a cv2_image. First contour is for the left eye"""
        contours = []
        gray_image = Utils.convert_to_gray_image(cv2_image)
        rects = self._face_detector(gray_image, 0)
        if len(rects) > 0:
            # only for the first face found
            shape = self._face_predictor(gray_image, rects[0])
            shape = face_utils.shape_to_np(shape)
            (left_eye_start,
             left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            (right_eye_start,
             right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
            contours.append(shape[left_eye_start:left_eye_end])
            contours.append(shape[right_eye_start:right_eye_end])
        return contours


face_detector = None
face_predictor = None
stuff_was_initialized = False


def initialize_stuff():
    global stuff_was_initialized, face_detector, face_predictor
    stuff_was_initialized = True
    face_detector = dlib.get_frontal_face_detector()
    face_predictor = dlib.shape_predictor(Config.face_landmarks_path)


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
        start, end = min(left_eye_start, right_eye_start), max(left_eye_end, right_eye_end)
        strip = shape[start:end]
        # get the upper left point, lower right point
        start = [min(strip, key = lambda x: x[0])[0], min(strip, key = lambda x: x[1])[1]]
        end = [max(strip, key = lambda x: x[0])[0], max(strip, key = lambda x: x[1])[1]]
        # go a little outside the bounding box, to capture more details
        distance = (end[0] - start[0], end[1] - start[1])
        # 20 percent more details on the X axis, 60% more details on the Y axis
        percents = [20, 60]
        for i in range (0, 2):
            start[i] -= int(percents[i]/100 * distance[i])
            end[i] += int(percents[i]/100 * distance[i])
        return cv2_image[start[1]:end[1], start[0]:end[0]]
    return None
