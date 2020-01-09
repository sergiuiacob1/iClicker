import numpy as np
import os
import joblib

from face_detector import get_eye_contours
from config import EYE_WIDTH, EYE_HEIGHT, data_directory_path, train_data_path
from utils import resize_cv2_image


def load_collected_data():
    data = []
    for r, _, files in os.walk(data_directory_path):
        for file in files:
            if file.endswith('.pkl'):
                current_data = joblib.load(os.path.join(r, file))
                data.extend(current_data)
    return data


def get_eye_images(data):
    """Returns a list of tuples from collected data. The returned list is not necessarily the same length, because some data can be useless.

    The result looks like: `[(left_eye_cv2_image, right_eye_cv2_image), ...]`
    """
    n = len(data)
    eye_images = [None] * n
    last_valid_image = 0

    for i in range(0, n):
        if i % 10 == 0:
            print(f'Processed eyes for {i}/{n} images')
        img = data[i].image
        eye_contours = get_eye_contours(img)
        if len(eye_contours) == 0 or len(eye_contours[0]) == 0 or len(eye_contours[1]) == 0:
            continue

        # I identified both eyes
        current_eye_images = []
        for eye_contour in eye_contours:
            x_min = min([x[0] for x in eye_contour])
            x_max = max([x[0] for x in eye_contour])
            y_min = min([x[1] for x in eye_contour])
            y_max = max([x[1] for x in eye_contour])
            resized_eye_image = resize_cv2_image(
                img[y_min:y_max, x_min:x_max], fixed_dim=(EYE_WIDTH, EYE_HEIGHT))
            current_eye_images.append(resized_eye_image)

        eye_images[last_valid_image] = tuple(current_eye_images)
        last_valid_image += 1

    eye_images = eye_images[:last_valid_image]
    return eye_images


def process_data(input_data):
    print('Extracting eye data...')
    data = get_eye_images(input_data)
    print('Normalizing...')
    normalize_data(data)
    # returning a list of tuples [(X, y), (X, y)] where X is a tuple with the eye images
    print ('Creating final train data...')
    processed_data = [(X, y) for X, y in zip (data, [x.horizontal for x in input_data])]
    return processed_data


def normalize_data(data):
    for i in range(0, len(data)):
        data[i] = (np.array(data[i][0])/255, np.array(data[i][1])/255)


def save_processed_data(data):
    os.makedirs(os.path.join(os.getcwd(), train_data_path), exist_ok=True)
    joblib.dump(data, os.path.join(
        os.getcwd(), train_data_path, 'processed_data.pkl'))


def main():
    # load the data
    print('Loading collected data...')
    data = load_collected_data()
    # process it
    print('Processing data...')
    processed_data = process_data(data)
    # save the result
    print('Saving processed data...')
    save_processed_data(processed_data)


if __name__ == '__main__':
    main()
