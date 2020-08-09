import os
import cv2
import numpy as np

class FeatureExtractor(object):
    def __init__(self, maxCorners=3000, minDistance=10):
        ### Define the number of grids to divide ###
        self.Gx = 8
        self.Gy = 6
        self.last = None ### For feature matching ###

        ### Initialize and ORB detector ###
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher()
        
        ### Intialize other variables ###
        self.maxCorners = maxCorners
        self.qualityLevel = 0.01
        self.minDistance = minDistance

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


        ### Detecting key points ###
        corners = cv2.goodFeaturesToTrack(frame, self.maxCorners, self.qualityLevel, self.minDistance)

        ### Extraction ###
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in corners]
        kps, des = self.orb.compute(frame, kps)
        self.last = {'kps' : kps, 'des' : des}

        ### Matching ###
        matches = None
        matching_pairs = []
        if(self.last is not None):
            # matches = self.bf.match(des, self.last['des'])
            ### Find matching points from the last frame ###
            matches = self.bf.knnMatch(des, self.last['des'], k=2)

            ### Since we got 2 best matches per DMatch object ###
            for m1, m2 in matches:
                ### if the first match is considerably closer ###
                if(m1.distance < 0.75 * m2.distance):
                    matching_pairs.append((kps[m1.queryIdx], self.last['kps'][m1.trainIdx]))

        return kps, des, matching_pairs
