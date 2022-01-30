import cv2
from cv2 import add
from matplotlib.pyplot import axis
import numpy as np
import math
from sklearn.cluster import KMeans

import json
import os
import glob

def get_mosift_features(cap):
    frame_idx = 0
    last_frame = 0
    original_resolution = 128
    resolution_step = 4
    s = 1
    sample_count = s + 3
    delta = np.power(2, 1/s)
    X = []

    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (128, 128))

            sift = cv2.SIFT_create()
            keypoints, des = sift.detectAndCompute(frame ,None)
            
            # ---------------------------------------------------------------------------- #
            #                               Describe Features                              #
            # ---------------------------------------------------------------------------- #
            exibitional = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if frame_idx > 0:
                flow = cv2.calcOpticalFlowFarneback(last_frame, frame, None, 0.5, 3, 3, 3, 5, 1.2, 0)

                descriptors = []
                for kp_idx, kp in enumerate(keypoints):
                    x = kp.pt[0]
                    y = kp.pt[1]

                    x = int(x)
                    y = int(y)

                    if(x < 16 or y < 16 or x > 110 or y > 110):
                        continue
                    
                    motion_histogram = []
                    mag_sum = 0
                    for i in range(-2, 2):
                        for j in range(-2, 2):
                            bin = [0, 0, 0, 0, 0, 0, 0, 0]
                            s1 = x + (i * 4)
                            e1 = x + (i * 4) + 4
                            s2 = y + (j * 4)
                            e2 = y + (j * 4) + 4
                            for fp in flow[s1:e1]:
                                for f in fp[s2:e2]:
                                    flow_x = f[0]
                                    flow_y = f[1]
                                    mag = math.sqrt(flow_x ** 2 + flow_y ** 2)
                                    mag_sum += mag
                                    angle = math.degrees(math.atan2(flow_y, flow_x))
                                    idx = int(angle / 45)
                                    if angle < 0:
                                        idx = int((360 + angle) / 45)
                                    bin[idx] += mag
                            motion_histogram.append(bin)
                    
                    if(mag_sum > 30):
                        exibitional = cv2.circle(exibitional, (x, y), 5, (255, 255, 0))
                        mh = np.array(motion_histogram, dtype=np.float32)
                        mh = np.reshape(mh, (128,))
                        d = np.array(des[kp_idx])
                        d = np.concatenate([d, mh])

                        descriptors.append(d)

                
                cv2.imshow('frame', exibitional)
                X = X + descriptors

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            last_frame = frame
            frame_idx += 1
            ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if(ms > 3000):
                break

        else:
            # cap.set(cv2.CAP_PROP_POS_MSEC, 0)
            # done processing the video
            break
    
    X = np.array(X)

    try:
        kmeans = KMeans(n_clusters=16, random_state=0).fit(X)
        centers = kmeans.cluster_centers_

        cap.release()
        cv2.destroyAllWindows()
        return centers
    except:
        return np.array([])

feature_array = []

def process(fname, i):
    for filename in glob.glob('./{fname}/*.avi'.format(fname=fname)):
        cap = cv2.VideoCapture(os.path.join(os.getcwd(), filename))
        features = get_mosift_features(cap)
        f = features.tolist()
        if len(f) > 0:
            feature_array.append(f)

    with open('{i}.json'.format(i=i), 'w', newline = '\n') as jsonfile:
        json.dump(feature_array, jsonfile)

i = 0
for filename in glob.glob('KTH/*'):
    process(filename, i)
    i += 1