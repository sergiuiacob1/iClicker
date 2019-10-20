from mouse_listener import MouseListener
from webcam_capturer import WebcamCapturer
import logging


class TrainingDataCollector:
    def __init__(self):
        self.mouseListener = MouseListener(self.onMouseClick)
        self.webcamCapturer = WebcamCapturer()
        logging.basicConfig(filename=("mouse_logs.txt"), level=logging.DEBUG,
                            format='%(asctime)s: %(message)s')

    def startCollecting(self):
        print('Started collecting training data...')
        self.mouseListener.startListening()
        self.webcamCapturer.startCapturing()

    def onMouseClick(self, x, y, button, pressed):
        if pressed:
            logging.info(f'{button} pressed at ({x}, {y})')
            a, webcamImage = self.webcamCapturer.getWebcamImage()
            print(a)
            
