import numpy as np
import os
import joblib
import re
import json
from cv2 import imread
import time

# My files
# TODO delete from config import ...
from config import EYE_WIDTH, EYE_HEIGHT, data_directory_path, train_data_path, WEBCAM_IMAGE_HEIGHT, WEBCAM_IMAGE_WIDTH
import config as Config
import src.face_detector as face_detector
from src.utils import resize_cv2_image, get_binary_thresholded_image, convert_to_gray_image, apply_function_per_thread, setup_logger
from src.data_object import DataObject

dp_logger = setup_logger('dp_logger', './logs/data_processing.log')

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
    # TODO split this work between processors
    st = time.time()
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
            try:
                session_items = json.load(f)
            except Exception as e:
                print(f'Could not process session {i}: {str(e)}')

        # TODO try to do this, but careful; test before and after
        # temp = [None] * len(session_items)
        # for (index, x) in enumerate(session_items):
        #     img_name = f"{i}_{x}.png"
        #     temp[index] = os.path.join(images_path, img_name)
        # # This function will load all the data in temp for the current session
        # # Each thread will load part of the data
        # apply_function_per_thread(temp, lambda x: imread(x))

        for x in session_items:
            img_name = f"{i}_{x}.png"
            img = imread(os.path.join(images_path, img_name))
            if img is None:
                continue
            mouse_position = session_items[x]["mouse_position"]
            data.append(DataObject(img, mouse_position, screen_size))
    dp_logger.info(f'Loading data took {time.time() - st} seconds')
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
        eye_contours = face_detector.FaceDetector().get_eye_contours(img)
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


def extract_faces(X):
    apply_function_per_thread(X, face_detector.extract_face)
    # TODO use the function above for the rest of the transformations. do before & after comparison and write comparison in report
    for i in range(0, len(X)):
        # X[i] = face_detector.extract_face(X[i])
        # resize this image
        X[i] = resize_cv2_image(X[i], fixed_dim=(
            Config.FACE_WIDTH, Config.FACE_HEIGHT))
        # also convert to grayscale, for now
        X[i] = convert_to_gray_image(X[i])
    return X


def normalize_data(data):
    # TODO watch out, this uses fixed values
    for i in range(0, len(data)):
        data[i] = np.true_divide(data[i], 255)

    # for i in range(0, len(data)):
    #     data[i] = (np.array(data[i][0])/255, np.array(data[i][1])/255)


def save_processed_data(data, name='train_data.pkl'):
    os.makedirs(os.path.join(os.getcwd(), train_data_path), exist_ok=True)
    joblib.dump(data, os.path.join(os.getcwd(), train_data_path, name))


def process_data_extract_faces(data):
    # right now, get data close to the corners
    data = [x for x in data if x.is_close_to_corner is True]
    print(f'Only selected corner data: {len(data)} items')

    # process it
    print('Processing data...')
    X = extract_faces([x.image for x in data])
    y = [[1 if i == x.square else 0 for i in range(0, 4)]
         for x in data]
    # it's possible that some faces weren't found, in which case they are None
    print(f'Selecting data for which faces were found. Before: {len(X)} items')
    X, y = np.array(X), np.array(y)
    faces_found = [True] * len(X)
    for (index, x) in enumerate(X):
        if x is None:
            faces_found[index] = False
    X = X[faces_found]
    y = y[faces_found]
    print(f'After: {len(X)} items')

    # normalize the data
    print('Normalising...')
    normalize_data(X)

    # save processed data
    print(f'Saving processed data: {len(X)} final items')
    save_processed_data((X, y), 'extracted_faces.pkl')


def main():
    # load the data
    print('Loading collected data...')
    data = load_collected_data()
    print(f'Loaded {len(data)} items')

    f = process_data_extract_faces

    start = time.time()
    f(data)
    s = f'Processing data with {f} took {time.time() - start} seconds for {len(data)} original items'
    print(s)
    dp_logger.info(s)

    # data = [x for x in data if x.is_close_to_corner is True]
    # print(f'Only selected corner data: {len(data)} items')

    # # process it
    # print('Processing data...')
    # X = process_images([x.image for x in data])
    # y = [[1 if i == x.square else 0 for i in range(0, 4)]
    #      for x in data]

    # # TODO this below only happens in some cases, so make sure it's okay to stick around
    # # this below can happen because some of the images might be useless, and therefore not returned in the "processing" part
    # if len(X) < len(y):
    #     y = y[:len(X)]
    # processed_data = (X, y)

    # save the result
    # print(f'Saving processed data: {len(processed_data[0])} final items')
    # save_processed_data(processed_data)


if __name__ == '__main__':
    main()
