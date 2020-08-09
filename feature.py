import os
import cv2
import numpy as np

class FeatureExtractor(object):
    def __init__(self):
        ### Define the number of grids to divide ###
        self.Gx = 8
        self.Gy = 6

        ### Initialize and ORB detector ###
        self.orb = cv2.ORB_create()
    
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
