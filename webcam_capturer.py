from cv2 import cv2
import time
import os
import threading
from multiprocessing import Process


class WebcamCapturer:
    def startCapturing(self):
        self.cam = cv2.VideoCapture(0)

    def getWebcamImage(self):
        if hasattr(self, 'cam') == False:
            self.startCapturing()
        ret, frame = self.cam.read()
        return ret, frame
        # except Exception as e:
        #     print (f'Exception: {str(e)}')
        #     print ('Maybe the webcam is not started?')
        #     return False, None


    # def saveCurrentWebcamImage(self, path):
    #     if hasattr(self, 'cam') == False:
    #         print('Webcam not started, cannot save current image!')
    #         return
    #     success, image = self.getWebcamImage()
    #     if success is not True:
    #         print('Could not read current image from webcam!')
    #         return

    #     os.makedirs(path, exist_ok=True)
    #     secondsSinceEpoch = time.time()
    #     cv2.imwrite(f'{path}/{secondsSinceEpoch}.png', image)

    def previewWebcam(self):
        windowName = 'Webcam Preview'
        fps = 60
        cv2.namedWindow(windowName)

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
    input('Done...')
    # web.startCapturing()
    # web.saveCurrentWebcamImage('data')
