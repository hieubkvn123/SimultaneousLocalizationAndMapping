import os
import cv2
import numpy as np

class FeatureExtractor(object):
    def __init__(self, maxCorners=3000):
        ### Define the number of grids to divide ###
        self.Gx = 8
        self.Gy = 6

        ### Initialize and ORB detector ###
        self.orb = cv2.ORB_create()
        
        ### Intialize other variables ###
        self.maxCorners = maxCorners
        self.qualityLevel = 0.01
        self.minDistance = 10

    ### Extracting key features using ORB ###
    def getKeyPoints(self, frame):
        Gx_size = frame.shape[1] // self.Gx
        Gy_size = frame.shape[0] // self.Gy

        key_points = []

        ### Divide the frame into chunks ###
        for ry in range(0, frame.shape[0], Gy_size):
            for rx in range(0, frame.shape[1], Gx_size):
                img_chunk = frame[ry:ry+Gy_size, rx:rx+Gx_size]
                
                ### Detect keypoints for every chunk ###
                kp = self.orb.detect(img_chunk, None)

                ### Convert keypoints' coords back to original coords of the big image ###
                for p in kp:
                    p.pt = (p.pt[0] + rx, p.pt[1] + ry)
                    key_points.append(p)

        return key_points

    def goodFeaturesToTrack(self, frame):
        ### Convert to gray if not gray ###
        if(len(frame.shape) == 3 and frame.shape[2] != 1):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners = cv2.goodFeaturesToTrack(frame, self.maxCorners, self.qualityLevel, self.minDistance)
        corners = np.int0(corners) # Parse the coords back to int 

        return corners
