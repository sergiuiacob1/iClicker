from cv2 import cv2
import time
import os
import threading
from multiprocessing import Process


# TODO make this a Singleton instead?
class WebcamCapturer:
    def __init__(self):
        self.webcam_lock = threading.Lock()

    def startCapturing(self):
        self.webcam_lock.acquire()
        self.cam = cv2.VideoCapture(0)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.webcam_lock.release()

    def stopCapturing(self):
        if hasattr(self, "cam") is False:
            return
        self.webcam_lock.acquire()
        self.cam.release()
        del self.cam
        self.webcam_lock.release()

    def webcam_is_started(self):
        return hasattr(self, 'cam') and self.cam.isOpened()

    def getWebcamImage(self, start_if_not_started=True):
        if self.webcam_is_started() is False:
            if start_if_not_started:
                self.startCapturing()
            else:
                return False, None
        self.webcam_lock.acquire()
        ret, frame = self.cam.read()
        self.webcam_lock.release()
        return ret, frame

    def previewWebcam(self):
        windowName = 'Webcam Preview'
        fps = 60
        cv2.namedWindow(windowName)

        if self.webcam_is_started() is False:
            self.startCapturing()

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
    # web.previewWebcam()
    web.startCapturing()
    web.getWebcamImage()
    web.stopCapturing()
    web.getWebcamImage()
    # web.startCapturing()
    # web.saveCurrentWebcamImage('data')
