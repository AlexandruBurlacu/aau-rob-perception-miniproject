import cv2
import numpy as np

import time


video = cv2.VideoCapture(0)

us = 0
begin = time.time()

abs_mean = 0

def preprocess(img):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # res_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    res_img = img / 255.
    for i in range(3):
        res_img[i, :, :] = (res_img[i, :, :] - mean[i]) / std[i]
    res_img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 10, 8)
    return res_img

def get_features(img):
    # hu moments
    bw_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hu_moments = cv2.HuMoments(cv2.moments(bw_image)).flatten()

    # means
    img_mean = np.mean(img)
    mean_b = np.mean(img[0, :, :])
    mean_g = np.mean(img[1, :, :])
    mean_r = np.mean(img[2, :, :])
    means_vec = [img_mean, mean_r, mean_g, mean_b]

    # orb # debug it
    orb = cv2.ORB()
    kp = orb.detect(bw_image, None)
    kp, orb_descriptors = orb.compute(bw_image, kp)

    return {"hu": hu_moments, "orb": orb_descriptors, "means": means_vec}

while True:

    us = us + 1

    check, frame = video.read()

    proc_frame = preprocess(frame)

    cv2.imshow("Capture", proc_frame)

    print(get_features(proc_frame))

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
dt = time.time() - begin

print(f"{us} frames elapsed in {dt} seconds")
video.release()
cv2.destroyAllWindows()
print("All is good, bye")

