import cv2
import os
import numpy as np
import random
import csv
import json
from math import sqrt


eye_radius = {}
eye_centers = {}
current_image = None


def dist(pos1, pos2):
    return sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


def click(event, x, y, flags, param):
    global eye_radius, current_image
    if event == cv2.EVENT_LBUTTONDOWN:
        eye_radius[current_image] = dist((x, y), eye_centers[current_image])


def read_eye_centers():
    global eye_centers
    with open('crowdpupil/results.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';')
        for row in spamreader:
            eye_centers[row[0]] = (int(row[1]), int(row[2]))


if __name__ == '__main__':
    read_eye_centers()
    window_name = 'eye image'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click)

    images = os.listdir('./crowdpupil/images')
    random.shuffle(images)

    for i in range(0, 50):
        current_image = images[i]
        img = cv2.imread('./crowdpupil/images/' + current_image)
        cv2.imshow(window_name, img)
        key = cv2.waitKey()
        if key == '27':
            break

    # save the eye_radius
    with open('crowdpupil/eye_radius.json', 'w') as fp:
        json.dump(eye_radius, fp)

    # print the average
    print (sum(eye_radius.values()) / len(eye_radius))
