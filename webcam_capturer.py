import cv2


class WebcamCapturer:
    def __init__(self):
        self.cam = cv2.VideoCapture(0)

    def startCapturing(self):
        ...

    def getWebcamImage(self):
        ret, frame = self.cam.read()
        return ret, frame

    def previewWebcam(self):
        with open('config.json') as configFile:
            config = json.load(configFile)

        cam = cv2.VideoCapture(0)
        cv2.namedWindow('iClicker')

        def saveImage():
            imgName = f'{config["dataDirectoryPath"]}/{time.time()}.png'
            cv2.imwrite(imgName, frame)
            print("{} written!".format(imgName))

        while True:
            ret, frame = cam.read()
            cv2.imshow("test", frame)
            if not ret:
                break
            k = cv2.waitKey(1)

            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k % 256 == 32:
                # SPACE pressed
                saveImage()

    def __del__(self):
        self.cam.release()
        cv2.destroyAllWindows()
