import joblib
import cv2
import random

which = 'eye_strips_3.pkl'
data = joblib.load(f'./train_data/{which}')

print(data[1][1])
print(data[0][0].shape)
for i in range (0, 4):
    summm = sum([1 for x in data[1] if x[i] == 1])
    print(f'Class {i} has {summm} instances')

# n = len(data[0])
# for i in range(0, 10):
#     r = random.randint(0, n - 1)
#     img = data[0][r]
#     print (data[1][r])
#     cv2.imshow(f'{i}', img)
#     cv2.waitKey(0)

# cv2.destroyAllWindows()
