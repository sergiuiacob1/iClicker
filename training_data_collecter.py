from mouse_listener import MouseListener
from webcam_capturer import WebcamCapturer


class TrainingDataCollector:
    def __init__(self):
        self.mouseListener = MouseListener(self.onMouseClick)

    def startCollecting(self):
        print ('Started collecting training data...')
        self.mouseListener.startListening()
        print ('DACA A AJUNS AICI E BINE BINE')

    def onMouseClick(self, x, y, button, pressed):
        if pressed:
            print ('BUTTON PRESSEEEEEEEEEED')
            logging.info(
                'Mouse clicked at ({0}, {1}) with {2}'.format(x, y, button))
