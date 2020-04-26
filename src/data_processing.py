import numpy as np
import os
import joblib
import re
import json
from cv2 import imread
import time
from multiprocessing import Pool

# My files
import config as Config
import src.face_detector as face_detector
from src.utils import resize_cv2_image, get_binary_thresholded_image, convert_to_gray_image, setup_logger, attach_logger_to_stdout
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
    st = time.time()
    data = []
    data_path = os.path.join(os.getcwd(), Config.data_directory_path)
    sessions_path = os.path.join(data_path, "sessions")
    images_path = os.path.join(data_path, "images")

    if os.path.exists(os.path.join(data_path, "sessions.json")) is False:
        return None

    with open(os.path.join(data_path, "sessions.json")) as f:
        sessions_info = json.load(f)

    for i in range(1, sessions_info["total_sessions"] + 1):
        session_info = sessions_info[f"session_{i}"]
        screen_size = session_info["screen_size"]

        with open(os.path.join(sessions_path, f"session_{i}.json")) as f:
            try:
                session_items = json.load(f)
            except Exception as e:
                dp_logger.info(f'Could not process session {i}: {str(e)}')

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
            data.append(DataObject(
                img, mouse_position, screen_size, Config.grid_size))
        
    dp_logger.info(f'Loading data took {time.time() - st} seconds')
    return data


def extract_thresholded_eyes(X):
    """This extracts the eyes from the images, converts the eyes to gray images, then applies a threshold to obtain a black and white image.

    Merges the images of the 2 eyes horizontally, into a single image."""
    for i in range(0, len(X)):
        eyes = face_detector.extract_eyes(X[i])
        # in case no eyes were detected
        if eyes is None or eyes[0] is None or eyes[1] is None:
            X[i] = None
            continue
        # resize each eye
        for j in range(0, 2):
            eyes[j] = resize_cv2_image(eyes[j], fixed_dim=(
                Config.EYE_WIDTH, Config.EYE_HEIGHT))
        # merge the eyes horizontally
        eyes = np.concatenate((eyes[0], eyes[1]), axis=1)
        # take only the left eye
        X[i] = eyes
        # threshold the eyes. this also converts the image to grayscale
        X[i] = get_binary_thresholded_image(X[i])
        # normalise
        X[i] = np.true_divide(X[i], 255)
    return np.array(X)


def extract_faces(X):
    """
    Does all the processing related to face extraction.

    Return a `np.array`."""
    for i in range(0, len(X)):
        X[i] = face_detector.extract_face(X[i])
        # in case no face was detected
        if X[i] is None:
            continue
        # resize this image
        X[i] = resize_cv2_image(X[i], fixed_dim=(
            Config.FACE_WIDTH, Config.FACE_HEIGHT))
        if X[i] is None:
            continue
        # also convert to grayscale, for now
        X[i] = convert_to_gray_image(X[i])
        # normalise
        X[i] = np.true_divide(X[i], 255)
    return np.array(X)


def extract_eye_strips(X):
    """This extracts eye strips from the images.

    Returns a `np.array`."""

    for i in range(0, len(X)):
        X[i] = face_detector.extract_eye_strip(X[i])
        # in case no face was detected
        if X[i] is None:
            continue
        # resize this image
        X[i] = resize_cv2_image(X[i], fixed_dim=(
            Config.EYE_STRIP_WIDTH, Config.EYE_STRIP_HEIGHT))
        if X[i] is None:
            continue
        # also convert to grayscale, for now
        X[i] = convert_to_gray_image(X[i])
        # normalise
        X[i] = np.true_divide(X[i], 255)
    return np.array(X)


def save_processed_data(data, name='train_data.pkl'):
    os.makedirs(os.path.join(
        os.getcwd(), Config.train_data_path), exist_ok=True)
    joblib.dump(data, os.path.join(os.getcwd(), Config.train_data_path, name))


def process_data(data, how_to_process_it):
    # # right now, get data close to the corners
    # data = [x for x in data if x.cell in [0, Config.grid_size - 1,
    #                                       Config.grid_size * (Config.grid_size - 1), Config.grid_size * Config.grid_size - 1]]
    # dp_logger.info(f'Only selected corner data: {len(data)} items')

    # process it
    dp_logger.info(f'Processing data using {how_to_process_it}')
    X = how_to_process_it([x.image for x in data])
    y = [[1 if i == x.cell else 0 for i in range(0, Config.grid_size * Config.grid_size)]
         for x in data]
    y = np.array(y)
    # it's possible that some instances couldn't be processed, therefore eliminate those
    dp_logger.info(f'Selecting data for which faces were found. Before: {len(X)} items')
    instances_success = [True] * len(X)
    for (index, x) in enumerate(X):
        if x is None:
            instances_success[index] = False
    X = X[instances_success]
    y = y[instances_success]
    dp_logger.info(f'After: {len(X)} items')

    assert (np.amax(X) <= 1), "Data isn't normalised"

    # save processed data
    dp_logger.info(f'Saving processed data: {len(X)} final items')
    if how_to_process_it == extract_eye_strips:
        name = f'eye_strips_{Config.grid_size}.pkl'
    elif how_to_process_it == extract_faces:
        name = f'extracted_faces_{Config.grid_size}.pkl'
    elif how_to_process_it == extract_thresholded_eyes:
        name = f'thresholded_eyes_{Config.grid_size}.pkl'
    save_processed_data((X, y), name)


def process_data_for_regression(data, how_to_process_it):
    # process it
    dp_logger.info(f'Processing data using {how_to_process_it}')
    X = how_to_process_it([x.image for x in data])
    y = [x.mouse_position for x in data]
    y = np.array(y)
    # it's possible that some instances couldn't be processed, therefore eliminate those
    dp_logger.info(f'Selecting data for which faces were found. Before: {len(X)} items')
    instances_success = [True] * len(X)
    for (index, x) in enumerate(X):
        if x is None:
            instances_success[index] = False
    X = X[instances_success]
    y = y[instances_success]
    dp_logger.info(f'After: {len(X)} items')

    assert (np.amax(X) <= 1), "Data isn't normalised"

    # save processed data
    dp_logger.info(f'Saving processed data: {len(X)} final items')
    if how_to_process_it == extract_eye_strips:
        name = f'eye_strips_regression.pkl'
    elif how_to_process_it == extract_faces:
        name = f'extracted_faces_regression.pkl'
    elif how_to_process_it == extract_thresholded_eyes:
        name = f'thresholded_eyes_regression.pkl'
    save_processed_data((X, y), name)


def main():
    # load the data
    dp_logger.info('Loading collected data...')
    data = load_collected_data()
    if data is None:
        dp_logger.info('No collected data found')
        raise Exception('No collected data found')
    dp_logger.info(f'Loaded {len(data)} items')

    # f = extract_thresholded_eyes
    # f = extract_faces
    f = extract_eye_strips

    start = time.time()
    # process_data(data, f)
    process_data_for_regression(data, f)
    s = f'Processing data with {f} took {time.time() - start} seconds for {len(data)} original items'
    dp_logger.info(s)

if __name__ == '__main__':
    attach_logger_to_stdout()
    main()
