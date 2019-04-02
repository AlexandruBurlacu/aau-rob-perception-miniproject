import cv2
import numpy as np
import pickle

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

def get_matcher():
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.BFMatcher_create() # index_params, search_params

    return flann

def match_classify(matcher, lookup_store, desc):
    MIN_MATCH_COUNT = 5
    # TODO: figure out how to find if image matches and is a known fruit
    for possible_match_desc, class_t in lookup_store:
        matches = matcher.knnMatch(desc, possible_match_desc, k=2)

        good_matches = [m[0] for m in matches
                        if len(m) == 2 and m[0].distance < m[1].distance * 0.75]
        
        if len(good_matches) > MIN_MATCH_COUNT:
            return True, class_t        

    return False, "undefined"

def load_lookup_store(fpath="lookup.db"):
    with open(fpath, "rb") as fptr:
        features = pickle.load(fptr)

    return features


if __name__ == "__main__":
    
    lookup_store = load_lookup_store()
    orb = cv2.ORB_create()
    flann = get_matcher()
    video = cv2.VideoCapture(0)

    us = 0
    begin = time.time()

    abs_mean = 0

    while True:

        us = us + 1

        check, frame = video.read()

        proc_frame = preprocess(frame)

        cv2.imshow("Capture", proc_frame)

        segments = get_segments(proc_frame, None)

        for idx, blob in enumerate(segments):
            feature = get_features(blob, orb)
            is_match, class_t = match_classify(flann, lookup_store, feature)
            if is_match:
                print(f"At frame #{us} blob #{idx} is a {class_t}")

        # TODO: visualize the matched objects on the whole image

        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    dt = time.time() - begin

    print(f"{us} frames elapsed in {dt} seconds")
    video.release()
    cv2.destroyAllWindows()
    print("All is good, bye")

