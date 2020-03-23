from cv2 import cv2
import time
from threading import Lock, Thread

# My files
import config as Config

last_image = None
last_success = False
cam = None
webcam_lock = Lock()
# Keep track of the number of requests the webcam has
# A request means that someone is using images from the camera
capture_requests = 0


def start_capturing():
    print('Starting web capture')
    global webcam_lock, cam, capture_requests
    capture_requests += 1
    # if the camera is already started, don't start it again
    if cam is not None:
        return
    webcam_lock.acquire()
    # lock acquired
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, Config.WEBCAM_IMAGE_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.WEBCAM_IMAGE_HEIGHT)
    # don't forget to release lock
    webcam_lock.release()
    # start updating
    Thread(target=update).start()


def update():
    global webcam_lock, last_image, last_success
    while True:
        webcam_lock.acquire()
        if cam is None or cam.isOpened() is False:
            webcam_lock.release()
            break
        last_success, last_image = cam.read()
        webcam_lock.release()
        time.sleep(1.0/Config.WEBCAM_FPS)


def stop_capturing():
    print('Stopping webcam capture')
    global cam, webcam_lock, capture_requests
    if cam is None:
        return
    webcam_lock.acquire()
    # lock acquired
    capture_requests -= 1
    if capture_requests <= 0:
        # nobody else is currently using the camera, so release it
        cam.release()
        # TODO are these below necessary
        cam = None
        cv2.destroyAllWindows()
    # don't forget to release lock
    webcam_lock.release()


def webcam_is_started():
    global cam
    return cam is not None and cam.isOpened()


def get_webcam_image():
    global last_success, last_image
    return last_success, last_image
