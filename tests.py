import joblib
import cv2
import random

data = joblib.load('./train_data/eye_strips.pkl')

n = len(data[0])
for i in range(0, 10):
    r = random.randint(0, n - 1)
    img = data[0][r]
    print (data[1][r])
    cv2.imshow(f'{i}', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
