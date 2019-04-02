import cv2
import numpy as np

import time

video = cv2.VideoCapture(0)

us = 0
begin = time.time()

abs_mean = 0

def get_sbd_params():
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 1500

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    return params

def preprocess(img):
    res_img = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 5, 7)
    return res_img


def draw_keypoints(vis, keypoints, color = (0, 255, 255)):
    for kp in keypoints:
        x, y = kp.pt
        cv2.circle(vis, (int(x), int(y)), 2, color)


def get_segments(img, detector):
    bw_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blobs = detector.detect(bw_image)

    draw_keypoints(bw_image, blobs, (0,0,255))
 
    # Show keypoints
    cv2.imshow("Keypoints", bw_image)
    cv2.waitKey(0)

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
sbd = cv2.SimpleBlobDetector_create(get_sbd_params())

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

