from cv2 import cv2
import time
import os
import threading
from multiprocessing import Process


class WebcamCapturer:
    def startCapturing(self):
        self.cam = cv2.VideoCapture(0)

    def webcamIsStarted(self):
        return hasattr(self, 'cam')

    def getWebcamImage(self):
        if self.webcamIsStarted() is False:
            self.startCapturing()
        ret, frame = self.cam.read()
        return ret, frame

    def previewWebcam(self):
        windowName = 'Webcam Preview'
        fps = 60
        cv2.namedWindow(windowName)

        if self.webcamIsStarted() is False:
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
    web.previewWebcam()
    # web.startCapturing()
    # web.saveCurrentWebcamImage('data')
