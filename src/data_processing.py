import numpy as np
import os
import joblib
import re
import json
from cv2 import imread

# My files
from config import EYE_WIDTH, EYE_HEIGHT, data_directory_path, train_data_path
from src.face_detector import FaceDetector
from src.utils import resize_cv2_image, get_binary_thresholded_image
from src.data_object import DataObject


# class DataCollectionSession:
#     def __init__(self, session_info, session_number):
#         self.screen_size = session_info["screen_size"]
#         self.data = self._get_data(session_number)

#     def _get_data(self, session_number):
#         data_path = os.path.join(os.getcwd(), data_directory_path)
#         sessions_path = os.path.join(data_path, "sessions")
#         images_path = os.path.join(data_path, "images")

#         with open(os.path.join(sessions_path, f"session_{session_number}.json")) as f:
#             session_items = json.load(f)

#         data = [None] * len(session_items)

#         for (index, x) in enumerate(session_items):
#             img_name = os.path.join(images_path, f'{session_number}_{x}.png')
#             img = imread(img_name)
#             mouse_position = session_items[x]["mouse_position"]
#             obj = build_data_object(img, mouse_position, self.screen_size)
#             data[index] = obj
#         return data


def load_collected_data():
    data = []
    data_path = os.path.join(os.getcwd(), data_directory_path)
    sessions_path = os.path.join(data_path, "sessions")
    images_path = os.path.join(data_path, "images")

    if os.path.exists(os.path.join(data_path, "sessions.json")) is False:
        return []

    with open(os.path.join(data_path, "sessions.json")) as f:
        sessions_info = json.load(f)

    for i in range(1, sessions_info["total_sessions"] + 1):
        session_info = sessions_info[f"session_{i}"]
        screen_size = session_info["screen_size"]

        with open(os.path.join(sessions_path, f"session_{i}.json")) as f:
            session_items = json.load(f)

        for x in session_items:
            img_name = f"{i}_{x}.png"
            img = imread(os.path.join(images_path, img_name))
            mouse_position = session_items[x]["mouse_position"]
            data.append(DataObject(img, mouse_position, screen_size))
    return data


def get_eye_images(data):
    """Returns a list of tuples from `data`.

    The returned list is not necessarily the same length, because some data can be useless.

    The result looks like: `[(left_eye_cv2_image, right_eye_cv2_image), ...]`
    """
    n = len(data)
    eye_images = [None] * n
    last_valid_image = 0

    for i in range(0, n):
        if i % 10 == 0:
            print(f'Processed eyes for {i}/{n} images')
        img = data[i]
        eye_contours = FaceDetector().get_eye_contours(img)
        if len(eye_contours) == 0 or len(eye_contours[0]) == 0 or len(eye_contours[1]) == 0:
            continue

        # I identified both eyes
        current_eye_images = []
        for eye_contour in eye_contours:
            x_min = min([x[0] for x in eye_contour])
            x_max = max([x[0] for x in eye_contour])
            y_min = min([x[1] for x in eye_contour])
            y_max = max([x[1] for x in eye_contour])

            eye_image = img[y_min:y_max, x_min:x_max]
            resized_eye_image = resize_cv2_image(
                eye_image, fixed_dim=(EYE_WIDTH, EYE_HEIGHT))
            resized_eye_image = get_binary_thresholded_image(resized_eye_image)
            current_eye_images.append(resized_eye_image)

        eye_images[last_valid_image] = tuple(current_eye_images)
        last_valid_image += 1

    print(f'Recognized eye images for {last_valid_image}/{n} images')
    eye_images = eye_images[:last_valid_image]
    return eye_images


# def process_data(input_data):
#     """
#     Returns a tuple of 2 items: (`X`, `y`)

#     Both `X` and `y` are lists of train instances.
#     """
#     corner_data = [x for x in input_data if x.is_close_to_corner is True]
#     print(f'Only selected {len(corner_data)} items (close_to_corner == True)')
#     print('Extracting eye data...')
#     X = get_eye_images(corner_data)
#     normalize_data(X)
#     y = [[1 if (i + 1) == x.square else 0 for i in range(0, 4)]
#          for x in corner_data]

#     # TODO this below only happens in some cases, so make sure it's okay to stick around
#     # It's possible that X is shorter than y
#     if len(X) < len(y):
#         y = y[:len(X)]

#     processed_data = (X, y)
#     return processed_data


def process_images(images):
    print('Extracting eye data...')
    X = get_eye_images(images)
    normalize_data(X)
    return X


def normalize_data(data):
    # TODO watch out, this uses fixed values
    for i in range(0, len(data)):
        data[i] = (np.array(data[i][0])/255, np.array(data[i][1])/255)


def save_processed_data(data):
    os.makedirs(os.path.join(os.getcwd(), train_data_path), exist_ok=True)
    joblib.dump(data, os.path.join(
        os.getcwd(), train_data_path, 'train_data.pkl'))


def main():
    # load the data
    print('Loading collected data...')
    data = load_collected_data()
    print(f'Loaded {len(data)} items')

    data = [x for x in data if x.is_close_to_corner is True]
    print(f'Only selected corner data: {len(data)} items')

    # process it
    print('Processing data...')
    X = process_images([x.image for x in data])
    y = [[1 if (i + 1) == x.square else 0 for i in range(0, 4)]
         for x in data]

    # TODO this below only happens in some cases, so make sure it's okay to stick around
    # this below can happen because some of the images might be useless, and therefore not returned in the "processing" part
    if len(X) < len(y):
        y = y[:len(X)]
    processed_data = (X, y)

    # save the result
    print('Saving processed data')
    save_processed_data(processed_data)


if __name__ == '__main__':
    main()
