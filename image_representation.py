import cv2
import numpy as np

import time

# TODO: feature fusion? using NN Autoencoders?
# TODO: Texture matching coefficient


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
    orb = cv2.ORB_create()
    kp, orb_descriptors = orb.detectAndCompute(bw_image, None)

    return {"hu": hu_moments, "orb": orb_descriptors, "means": means_vec}

if __name__ == "__main__":

    video = cv2.VideoCapture(0)

    us = 0
    begin = time.time()

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

