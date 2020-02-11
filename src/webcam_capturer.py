from cv2 import cv2
import time
import os
import threading

import config as Config


# TODO make this a Singleton

# TODO better idea: keep a `last_captured_image` and update that every *FPS* if the webcam is on (webcam_on_requests > 0)
class WebcamCapturer:
    def __init__(self):
        self.webcam_lock = threading.Lock()

    def start_capturing(self):
        self.webcam_lock.acquire()
        self.cam = cv2.VideoCapture(0)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, Config.WEBCAM_IMAGE_WIDTH)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.WEBCAM_IMAGE_HEIGHT)
        self.webcam_lock.release()

    def stop_capturing(self):
        if hasattr(self, "cam") is False:
            return
        self.webcam_lock.acquire()
        self.cam.release()
        del self.cam
        self.webcam_lock.release()

    def webcam_is_started(self):
        return hasattr(self, 'cam') and self.cam.isOpened()

    def get_webcam_image(self, start_if_not_started=True):
        if self.webcam_is_started() is False:
            if start_if_not_started:
                self.start_capturing()
            else:
                return False, None
        self.webcam_lock.acquire()
        try:
            ret, frame = self.cam.read()
        except:
            ret, frame = False, None
        finally:
            self.webcam_lock.release()
            
        return ret, frame

    def preview_webcam(self):
        windowName = 'Webcam Preview'
        fps = 60
        cv2.namedWindow(windowName)

        if self.webcam_is_started() is False:
            self.start_capturing()

        while True:
            ret, frame = self.cam.read()
            cv2.imshow(windowName, frame)
            if not ret:
                break
            k = cv2.waitKey(1000//fps)

            if k % 256 == 27:
                # ESC pressed
                print('Escape hit, closing...')
                break
        cv2.destroyWindow(windowName)

    def __del__(self):
        if hasattr(self, 'cam'):
            self.cam.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    web = WebcamCapturer()
    # web.preview_webcam()
    web.start_capturing()
    web.get_webcam_image()
    web.stop_capturing()
    web.get_webcam_image()
    # web.start_capturing()
    # web.saveCurrentWebcamImage('data')
