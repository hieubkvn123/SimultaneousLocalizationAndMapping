import os
import cv2

from feature import FeatureExtractor
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-v", '--video', help="Input video path", required=True)
args = vars(parser.parse_args())

video = cv2.VideoCapture(args['video'])

### minDistance and maxCorners for cv2.goodFeaturesToTrack ###
extractor = FeatureExtractor(maxCorners=5000,minDistance=3)


def processFrame(frame):
    ### Convert to gray (Somehow computationally more efficient) ###
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # keypoints = extractor.getKeyPoints(frame)


    '''
    for p in keypoints:
        ### Basically round key points to integers ###
        x, y = map(lambda x : int(round(x)), p.pt)
        cv2.circle(frame, (x,y), color=(0,255,0), radius = 3)
    '''
    corners = extractor.goodFeaturesToTrack(frame)

    for i in corners:
        x, y = i.ravel() 
        cv2.circle(frame, (x,y), color=(0,255,0), radius = 3)

    return frame

if __name__ == '__main__':
    while(True):
        ret, frame = video.read()
        
        if(ret):
            frame = processFrame(frame)
        else:
            break
            print("[INFO] Failed to capture frame .. ")

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if(key == ord('q')):
            video.release()
            cv2.destroyAllWindows()
            break

video.release()
cv2.destroyAllWindows()
