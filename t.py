import numpy as np
import cv2 as cv

frame = np.zeros((32, 32, 3), np.uint8)
frame[16, 0:31, :] = 255
ret, buf = cv.imencode("toto.jpg", frame)
bufjpg = bytearray(buf)
fs = open("toto.jpg", "wb")
fs.write(bufjpg)
print(buf[0:15].tostring())
img = cv.imdecode(buf, cv.IMREAD_COLOR)
cv.imshow("img", img)
cv.waitKey(0)
