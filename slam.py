import os
import cv2

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-v", '--video', help="Input video path", required=True)
args = vars(parser.parse_args())

video = cv2.VideoCapture(args['video'])

def processFrame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ### Preprocessing steps ###

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
