import cv2
import numpy as np

import multiprocessing as mp

import time


def image_capture_proc(video, out_q):
    # while True:
    check, frame = video.read()
    out_q.put(frame)


def preprocess_proc(inp_q, out_q):
    # while not inp_q.empty():
    img = inp_q.get()

    res_img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 10, 8)

    out_q.put(res_img)


def feature_ext_proc(inp_q, out_q):
    # while not inp_q.empty():
    img = inp_q.get()

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

    out_q.put({"hu": hu_moments, "orb": orb_descriptors, "means": means_vec})


if __name__ == "__main__":
    img_capt_queue = mp.Queue(3)
    prep_queue = mp.Queue(3)
    feat_ext_queue = mp.Queue(3)

    video_inp = cv2.VideoCapture(0)

    p1 = mp.Process(target=image_capture_proc, args=(video_inp, img_capt_queue))
    p2 = mp.Process(target=preprocess_proc, args=(img_capt_queue, prep_queue))
    p3 = mp.Process(target=feature_ext_proc, args=(prep_queue, feat_ext_queue))

    [p.start() for p in (p1, p2, p3)]

    while True:
        # image_capture_proc(video_inp, img_capt_queue)
        print(feat_ext_queue.get())

        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    [p.join() for p in (p1, p2, p3)]
