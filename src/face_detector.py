import dlib
from imutils import face_utils
import src.config as Config
import src.utils as Utils

# TODO make a singleton here with lazy loading and threaded responses


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
