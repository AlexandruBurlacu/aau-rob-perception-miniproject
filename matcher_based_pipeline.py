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
    # ORB
    bw_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, orb_descriptors = orb.detectAndCompute(bw_image, None)

    return orb_descriptors

def get_flann():
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    return flann

def match_classify(matcher, lookup_store, desc):
    # TODO: figure out how to find if image matches and is a known fruit
    for possible_match_desc in lookup_store:
        matches = flann.knnMatch(desc, possible_match_desc, k=2)

        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in xrange(len(matches))]

        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matchesMask[i] = [1, 0]

    return False, "undefined"

def load_lookup_store():
    return None

lookup_store = load_lookup_store()
orb = cv2.ORB_create()
flann = get_flann()
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
        feature = get_features(blob, orb)
        is_match, class_t = match_classify(flann, lookup_store, feature)
        if is_match:
            print(class_t)

    # TODO: visualize the matched objects on the whole image

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

dt = time.time() - begin

print(f"{us} frames elapsed in {dt} seconds")
video.release()
cv2.destroyAllWindows()
print("All is good, bye")

