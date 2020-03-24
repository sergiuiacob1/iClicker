import joblib
import cv2

data = joblib.load('./train_data/extracted_faces.pkl')
img = data[0][100]

cv2.imshow('aa', img)
cv2.waitKey(0)
cv2.destroyAllWindows()