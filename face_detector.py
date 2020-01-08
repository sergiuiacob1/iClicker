import dlib
from imutils import face_utils
import config as Config
import utils as Utils

face_detector = dlib.get_frontal_face_detector()
face_predictor = dlib.shape_predictor(Config.face_landmarks_path)

# TODO make a singleton here with lazy loading and threaded responses


def get_eye_contours(cv2_image):
    """Returns eye contours from a cv2_image. First contour is for the left eye"""
    contours = []
    gray_image = Utils.convert_to_gray_image(cv2_image)
    rects = face_detector(gray_image, 0)
    if len(rects) > 0:
        # only for the first face
        shape = face_predictor(gray_image, rects[0])
        shape = face_utils.shape_to_np(shape)
        (left_eye_start,
         left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (right_eye_start,
         right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        contours.append(shape[left_eye_start:left_eye_end])
        contours.append(shape[right_eye_start:right_eye_end])
    return contours
