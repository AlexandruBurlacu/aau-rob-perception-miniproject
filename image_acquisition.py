import cv2
import numpy as np

import time


video = cv2.VideoCapture(0)
us = 0
begin = time.time()
while True:

    us = us + 1

    check, frame = video.read()

    cv2.imshow("Capture", frame)
    key = cv2.waitKey(1)

    if key == ord("q"):
        break
dt = time.time() - begin

print(f"{us} frames elapsed in {dt} seconds")
video.release()
cv2.destroyAllWindows()
print("All is good, bye")

