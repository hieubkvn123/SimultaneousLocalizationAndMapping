import os
import cv2

from feature import FeatureExtractor
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-v", '--video', help="Input video path", required=True)
args = vars(parser.parse_args())

video = cv2.VideoCapture(args['video'])

### minDistance and maxCorners for cv2.goodFeaturesToTrack ###
extractor = FeatureExtractor(maxCorners=3000,minDistance=3)


def processFrame(frame):
    ### Convert to gray (Somehow computationally more efficient) ###
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # keypoints = extractor.getKeyPoints(frame)


    '''
    for p in keypoints:
        ### Basically round key points to integers ###
        x, y = map(lambda x : int(round(x)), p.pt)
        cv2.circle(frame, (x,y), color=(0,255,0), radius = 3)
    '''
    kps, des, matches = extractor.goodFeaturesToTrack(frame)

    if(matches is None):
        return

    for p in kps:
        ### Basically round key points to integers ###
        x, y = map(lambda x : int(round(x)), p.pt)
        cv2.circle(frame, (x,y), color=(0,255,0), radius = 3)
 
    for pt1, pt2 in matches:
        x1, y1 = map(lambda x : int(round(x)), pt1.pt)
        x2, y2 = map(lambda x : int(round(x)), pt2.pt)

        cv2.line(frame, (x1, y1), (x2, y2), (0,0,255), thickness=2)

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
