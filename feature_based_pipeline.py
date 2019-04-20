import cv2
import numpy as np

import time


def preprocess(img):
    res_img = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 5, 7)
    return res_img


def get_segments(img, detector):
    bw_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return [img]

def get_features(img, orb):
    # hu moments
    bw_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hu_moments = cv2.HuMoments(cv2.moments(bw_image)).flatten()

    # means
    img_mean = np.mean(img)
    mean_b = np.mean(img[0, :, :])
    mean_g = np.mean(img[1, :, :])
    mean_r = np.mean(img[2, :, :])
    means_vec = [img_mean, mean_r, mean_g, mean_b]

    kp, orb_descriptors = orb.detectAndCompute(bw_image, None)

    return {"hu": hu_moments, "orb": orb_descriptors, "means": means_vec}


orb = cv2.ORB_create()

video = cv2.VideoCapture(0)

us = 0
begin = time.time()

abs_mean = 0

while True:

    us = us + 1

    check, frame = video.read()

    proc_frame = preprocess(frame)

    cv2.imshow("Capture", proc_frame)

    segments = get_segments(proc_frame, sbd)

    for blob in segments:
        print(get_features(blob, orb))

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
dt = time.time() - begin

print(f"{us} frames elapsed in {dt} seconds")
video.release()
cv2.destroyAllWindows()
print("All is good, bye")

