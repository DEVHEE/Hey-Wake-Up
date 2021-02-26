"""
Hey-Wake-Up
COPYRIGHT © 2021 KIM DONGHEE, KIM HYUNSANG. ALL RIGHTS RESERVED.
"""

# import modules
import time
import cv2
import dlib
import argparse
import numpy as np
import imutils
from imutils.video import VideoStream
from imutils import face_utils
from scipy.spatial import distance
from PIL import ImageFont, ImageDraw, Image

# import custom module
from gaze_tracking import GazeTracking

# set directory
landmark_dir = "./data/model/landmark/"
cascade_dir = "./data/model/cascade/"

# parse the arguments
'''
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True, help="set CascadeClassifier file in '"'./data/model/cascade'")
ap.add_argument("-w", "--webcam", type=int, default=0, help="set camera ID")
args = vars(ap.parse_args())
'''

# initialize models
print("[INFO] Initialize models...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(landmark_dir + "shape_predictor_68_face_landmarks.dat")
cascade = cv2.CascadeClassifier(cascade_dir + "haarcascade_frontalface_alt2.xml")  # cascade = cv2.CascadeClassifier(cascade_dir + args["cascade"])
print("Check")

# initialize camera id
print("[INFO] Set camera device...")
vid = VideoStream(0).start()  # vs = VideoStream(usePiCamera=args["webcam"] > 0).start()
time.sleep(1.0)
print("Check")

print("[INFO] Start detection system")


# def eyelid calc
def calc_lid(eye):
    s = distance.euclidean(eye[1], eye[5])
    t = distance.euclidean(eye[2], eye[4])
    u = distance.euclidean(eye[0], eye[3])
    lid_ear = (s + t) / (2.0 * u)
    return round(lid_ear, 3)


def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)


# set GazeTracking
gaze = GazeTracking()

# loop per frame
while True:
    frame = vid.read()
    frame = imutils.resize(frame, width=1000)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # count tick for fps
    tick = cv2.getTickCount()

    # detect face with cascade
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(100, 100))

    # draw detected face
    if len(faces) == 1:
        x, y, w, h = faces[0, :]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    '''
    gaze.refresh(frame)

    gazeFrame = gaze.annotated_frame()
    text = ""

    if gaze.is_blinking():
        text = "깜빡임\n\n왼쪽 눈동자 좌표: \n\n오른쪽 눈동자 좌표: "
    elif gaze.is_right():
        text = "오른쪽 응시\n\n왼쪽 눈동자 좌표: \n\n오른쪽 눈동자 좌표: "
    elif gaze.is_left():
        text = "왼쪽 응시\n\n왼쪽 눈동자 좌표: \n\n오른쪽 눈동자 좌표: "
    elif gaze.is_center():
        text = "정면 응시\n\n왼쪽 눈동자 좌표: \n\n오른쪽 눈동자 좌표: "
    else:
        text = "눈동자 인식 불가"

    pill_image = Image.fromarray(cv2.cvtColor(gazeFrame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pill_image)
    draw.text((50, 50), text, font=ImageFont.truetype('NanumGothic-Bold.ttf', 30), fill=(0, 255, 0))
    frame = cv2.cvtColor(np.array(pill_image), cv2.COLOR_RGB2BGR)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()

    cv2.putText(frame, str(left_pupil), (280, 135), cv2.FONT_HERSHEY_PLAIN, 2.2, (0, 255, 0), 2)
    cv2.putText(frame, str(right_pupil), (305, 195), cv2.FONT_HERSHEY_PLAIN, 2.2, (0, 255, 0), 2)
    '''

    # detect face with dlib
    rects = detector(gray, 0)

    # loop per face detections
    for rect in rects:
        mark = predictor(gray, rect)

        # set left eye point
        L_left_point = (mark.part(36).x, mark.part(36).y)
        L_right_point = (mark.part(39).x, mark.part(39).y)
        L_center_top = midpoint(mark.part(37), mark.part(38))
        L_center_bottom = midpoint(mark.part(41), mark.part(40))

        # set right eye point
        R_left_point = (mark.part(42).x, mark.part(42).y)
        R_right_point = (mark.part(45).x, mark.part(45).y)
        R_center_top = midpoint(mark.part(43), mark.part(44))
        R_center_bottom = midpoint(mark.part(47), mark.part(46))

        # set numpy mark
        mark = face_utils.shape_to_np(mark)

        # draw landmark
        for (x, y) in mark:
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

        # draw left lid line
        cv2.line(frame, L_left_point, L_right_point, (0, 255, 0), 1)
        cv2.line(frame, L_center_top, L_center_bottom, (0, 255, 0), 1)

        # draw right lid line
        cv2.line(frame, R_left_point, R_right_point, (0, 255, 0), 1)
        cv2.line(frame, R_center_top, R_center_bottom, (0, 255, 0), 1)

        # calc left lid ear
        left_lid_ear = calc_lid(mark[42:48])
        cv2.putText(frame, "LEFT LID EAR:{} ".format(round(left_lid_ear, 3)), (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

        # calc right lid ear
        right_lid_ear = calc_lid(mark[36:42])
        cv2.putText(frame, "RIGHT LID EAR:{} ".format(round(right_lid_ear, 3)), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

        # notice wake up
        if (left_lid_ear + right_lid_ear) < 0.40:
            cv2.putText(frame, "Hey, Wake Up!", (10, 180), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3, 1)

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - tick)
    cv2.putText(frame, "FPS:{} ".format(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2, cv2.LINE_AA)

    # show frames
    cv2.imshow("Hey, Wake Up!", frame)
    key = cv2.waitKey(1) & 0xFF

    # exit with "q"
    if key == ord("q"):
        break

# finish
cv2.destroyAllWindows()
vid.stop()