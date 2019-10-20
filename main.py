import cv2
import time

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

def saveImage():
    imgName = f'{time.time()}.png'
    cv2.imwrite(imgName, frame)
    print("{} written!".format(imgName))

while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        saveImage()
        

cam.release()

cv2.destroyAllWindows()

