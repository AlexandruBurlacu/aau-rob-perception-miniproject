import cv2
import numpy as np
import pickle

import time


def preprocess(img):
    res_img = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 5, 7)
    return res_img


def viewImage(image):
    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    cv2.imshow('Display', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_segments(img, detector):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    red_low1 = np.array([0, 100, 100])
    red_high1 = np.array([10, 255, 255])

    red_low2 = np.array([165, 100, 100])
    red_high2 = np.array([179, 255, 255])

    yellow_low = np.array([20, 100, 100])
    yellow_high = np.array([30, 255, 255])

    green_low = np.array([45 , 100, 100])
    green_high = np.array([75, 255, 255])

    imgs = []

    for lo, hi in (red_low1, red_high1), (red_low2, red_high2), (green_low, green_high), (yellow_low, yellow_high):

        curr_mask = cv2.inRange(hsv_img, lo, hi)
        hsv_img[curr_mask > 0] = (hi,)

        RGB_again = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
        gray = cv2.cvtColor(RGB_again, cv2.COLOR_RGB2GRAY)

        ret, threshold = cv2.threshold(gray, 90, 255, 0)

        kernel = np.ones((5,5), np.uint8)
        closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

        imgs.append(cv2.bitwise_and(img, img, mask=mask))

    return imgs


def get_features(img, orb):
    # ORB
    bw_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, orb_descriptors = orb.detectAndCompute(bw_image, None)

    return orb_descriptors


def get_matcher():
    matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)

    return matcher


def match_classify(matcher, lookup_store, desc):
    MIN_MATCH_COUNT = 5
    for possible_match_desc, class_t in lookup_store:
        matches = matcher.match(desc, possible_match_desc)

        good_matches = [m for m in matches
                        if m.distance < 0.7]
         
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
    matcher = get_matcher()
    video = cv2.VideoCapture(0)

    iters = 0
    begin = time.time()

    while True:

        iters += 1

        check, frame = video.read()

        proc_frame = preprocess(frame)

        cv2.imshow("Capture", proc_frame)

        segments = get_segments(proc_frame, None)

        for idx, blob in enumerate(segments):
            feature = get_features(blob, orb)
            is_match, class_t = match_classify(matcher, lookup_store, feature)
            if is_match:
                print(f"At frame #{iters} blob #{idx} is a {class_t}")

        # TODO: visualize the matched objects on the whole image

        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    dt = time.time() - begin

    print(f"{iters} frames elapsed in {dt} seconds")
    video.release()
    cv2.destroyAllWindows()
    print("All is good, bye")

