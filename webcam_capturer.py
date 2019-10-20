import cv2
import time
import os


class WebcamCapturer:
    def startCapturing(self):
        self.cam = cv2.VideoCapture(0)

    def getWebcamImage(self):
        ret, frame = self.cam.read()
        return ret, frame

    def saveCurrentWebcamImage(self, path):
        if hasattr(self, 'cam') == False:
            print('Webcam not started, cannot save current image!')
            return
        success, image = self.getWebcamImage()
        if success is not True:
            print('Could not read current image from webcam!')
            return

        os.makedirs(path, exist_ok=True)
        secondsSinceEpoch = time.time()
        cv2.imwrite(f'{path}/{secondsSinceEpoch}.png', image)
        # cv2.imwrite('ceva.png', image)

    def previewWebcam(self):
        # TODO this should be async?
        cv2.namedWindow('iClicker')

        while True:
            ret, frame = self.cam.read()
            cv2.imshow("WHAT IS THIS", frame)
            if not ret:
                break
            k = cv2.waitKey(1)

            if k % 256 == 27:
                # ESC pressed
                print('Escape hit, closing...')
                break

    def __del__(self):
        if hasattr(self, 'cam'):
            self.cam.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    web = WebcamCapturer()
    web.startCapturing()
    web.saveCurrentWebcamImage('data')
