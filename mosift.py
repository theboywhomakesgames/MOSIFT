import cv2
import numpy as np
import scipy.ndimage.filters as fi
from skimage.feature import peak_local_max
import math

def gkern(kernlen=21, nsig=3):
    inp = np.zeros((kernlen, kernlen))
    inp[kernlen//2, kernlen//2] = 1
    return fi.gaussian_filter(inp, nsig)

cap = cv2.VideoCapture("./KTH/boxing.zip_dir/person01_boxing_d1_uncomp.avi")
frame_idx = 0
last_frame = 0
original_resolution = 128
resolution_step = 4
s = 1
sample_count = s + 3
delta = np.power(2, 1/s)

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:
        # Prepare
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (original_resolution, original_resolution))
        octaves = []
        agg_test = []

        # ---------------------------------------------------------------------------- #
        #                             DoG for First Octave                             #
        # ---------------------------------------------------------------------------- #
        # create different levels of gaussians
        intervals = []
        for i in range(sample_count):
            kernel = gkern(3, delta * i)
            g_frame = cv2.filter2D(frame, -1, kernel)
            intervals.append(g_frame)
        
        # DoG
        dogs = []
        aggregated = np.zeros(frame.shape)
        for i in range(1, sample_count):
            diff = cv2.absdiff(intervals[i - 1], intervals[i])
            if i == 1:
                agg_test = diff
            dogs.append(diff)

        # create an octave
        octaves.append(dogs)
        
        # ---------------------------------------------------------------------------- #
        #                                Find Keypoints                                #
        # ---------------------------------------------------------------------------- #
        peaks_array = []
        for dog in octaves:
            peaks = peak_local_max(dog)
            peaks_array.append(peaks)

        mask = np.zeros_like(frame)

        # ---------------------------------------------------------------------------- #
        #                                 Temporal Part                                #
        # ---------------------------------------------------------------------------- #
        if frame_idx > 0:
            diff = cv2.absdiff(last_frame, frame)
            o = 0
            for peaks in peaks_array:
                for idx in peaks:
                    x = idx[0]
                    y = idx[1]
                    flag = False

                    pixel_value = octaves[o][x][y]

                    # Check local maxima against other octaves
                    dog_idx = -1
                    for dog in octaves:
                        dog_idx += 1
                        if(dog_idx == o):
                            continue

                        if(dog[x][y] > pixel_value):
                            flag = True
                            break
                    
                    if(flag):
                        break

                    if(diff[x][y] > 50):
                        mask[x][y] += 255

                cv2.imshow('features', agg_test)
                cv2.imshow('frame', frame)
                o += 1

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        last_frame = frame
        frame_idx += 1

    else:
        cap.set(cv2.CAP_PROP_POS_MSEC, 0)

cap.release()
cv2.destroyAllWindows()