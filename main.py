"""
Hey-Wake-Up
COPYRIGHT © 2021 KIM DONGHEE. ALL RIGHTS RESERVED.
"""

# import modules
import time
import math
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
from data.library.gaze_tracking import GazeTracking

# set directory
landmark_dir = "./data/model/landmark/"
cascade_dir = "./data/model/cascade/"

# parse the arguments
'''
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--front", required=True, help="set FrontalFace CascadeClassifier file in '"'./data/model/cascade'")
ap.add_argument("-w", "--webcam", type=int, default=0, help="set camera ID")
args = vars(ap.parse_args())
'''

# initialize models
print("[INFO] Initialize models...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(landmark_dir + "shape_predictor_68_face_landmarks.dat")
eye_cascade = cv2.CascadeClassifier(cascade_dir + "haarcascade_eye.xml")
front_cascade = cv2.CascadeClassifier(cascade_dir + "haarcascade_frontalface_alt2.xml")  # front_cascade = cv2.CascadeClassifier(cascade_dir + args["front"])
print("Check")

# initialize camera id
print("[INFO] Starting camera device...")
vid = VideoStream(0).start()  # vid = VideoStream(usePiCamera=args["webcam"] > 0).start()
print("[INFO] 2 Seconds left...")
time.sleep(1.0)
print("[INFO] 1 Second left...")
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

# set value container
angle_ab_list = []
L_LR_Cx_list = []
L_LR_Cy_list = []
R_LR_Cx_list = []
R_LR_Cy_list = []

frame_count = 0

# loop per frame
while True:
    print("==============================")
    print("angle: ", angle_ab_list)
    print("Lx: ", L_LR_Cx_list)
    print("Ly: ", L_LR_Cy_list)
    print("Rx: ", R_LR_Cx_list)
    print("Ry: ", R_LR_Cy_list)
    print("- - - - - - - - - - - - - - - -")

    frame = vid.read()
    frame = imutils.resize(frame, width=1000)

    # re-set frame with angle
    if frame_count != 0 and (R_LR_Cy_list[0] - L_LR_Cy_list[0]) > 0:
        rotate_frame = imutils.rotate(frame, angle_ab_list[0], ((L_LR_Cx_list[0] + R_LR_Cx_list[0]) // 2, (L_LR_Cy_list[0] + R_LR_Cy_list[0]) // 2))
        gray = cv2.cvtColor(rotate_frame, cv2.COLOR_BGR2GRAY)
    elif frame_count != 0 and (R_LR_Cy_list[0] - L_LR_Cy_list[0]) < 0:
        rotate_frame = imutils.rotate(frame, -angle_ab_list[0], ((L_LR_Cx_list[0] + R_LR_Cx_list[0]) // 2, (L_LR_Cy_list[0] + R_LR_Cy_list[0]) // 2))
        gray = cv2.cvtColor(rotate_frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # count tick for fps
    tick = cv2.getTickCount()

    '''[TEMP CODE] 1'''

    '''[TEMP CODE] 0'''

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

        # set point center
        # set point center
        L_LR_Cx = abs((mark.part(36).x + mark.part(39).x) // 2)
        L_LR_Cy = abs((mark.part(36).y + mark.part(39).y) // 2)

        L_TB_Cx = abs((mark.part(38).x + mark.part(40).x) // 2)
        L_TB_Cy = abs((mark.part(38).y + mark.part(40).y) // 2)

        R_LR_Cx = abs((mark.part(42).x + mark.part(45).x) // 2)
        R_LR_Cy = abs((mark.part(42).y + mark.part(45).y) // 2)

        R_TB_Cx = abs((mark.part(44).x + mark.part(46).x) // 2)
        R_TB_Cy = abs((mark.part(44).y + mark.part(46).y) // 2)

        # save pos value to container
        L_LR_Cx_list.append(L_LR_Cx)
        L_LR_Cy_list.append(L_LR_Cy)
        R_LR_Cx_list.append(R_LR_Cx)
        R_LR_Cy_list.append(R_LR_Cy)

        # set numpy mark
        mark = face_utils.shape_to_np(mark)

        # draw landmark
        for (x, y) in mark:
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

        # draw left lid line
        cv2.line(frame, L_left_point, L_right_point, (0, 255, 0), 2)
        cv2.line(frame, L_center_top, L_center_bottom, (0, 255, 0), 2)

        # draw right lid line
        cv2.line(frame, R_left_point, R_right_point, (0, 255, 0), 2)
        cv2.line(frame, R_center_top, R_center_bottom, (0, 255, 0), 2)

        # draw eyes
        cv2.circle(frame, (L_LR_Cx, L_TB_Cy), 23, (255, 0, 0), 2)
        cv2.circle(frame, (R_LR_Cx, R_TB_Cy), 23, (255, 0, 0), 2)

        # calc left lid ear
        left_lid_ear = calc_lid(mark[42:48])
        cv2.putText(frame, "LEFT LID EAR:{} ".format(round(left_lid_ear, 3)), (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

        # calc right lid ear
        right_lid_ear = calc_lid(mark[36:42])
        cv2.putText(frame, "RIGHT LID EAR:{} ".format(round(right_lid_ear, 3)), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

        # notice wake up
        if (left_lid_ear + right_lid_ear) < 0.40:
            cv2.putText(frame, "Hey, Wake Up!", (10, 180), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3, 1)

    # detect eyes with eye_cascade
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)

    # clear container
    del L_LR_Cx_list[:-1]
    del L_LR_Cy_list[:-1]
    del R_LR_Cx_list[:-1]
    del R_LR_Cy_list[:-1]

    print("Lx: ", L_LR_Cx_list)
    print("Ly: ", L_LR_Cy_list)
    print("Rx: ", R_LR_Cx_list)
    print("Ry: ", R_LR_Cy_list)

    # draw detected eyes
    index = 0
    for eye_x, eye_y, eye_w, eye_h in eyes:

        # set eye triangle
        side_a = math.sqrt((R_LR_Cx_list[0] - L_LR_Cx_list[0])**2 + (R_LR_Cy_list[0] - L_LR_Cy_list[0]))
        side_b = R_LR_Cx_list[0] - L_LR_Cx_list[0]
        side_c = R_LR_Cy_list[0] - L_LR_Cy_list[0]
        cos_ab = (side_a**2 + side_b**2 - side_c**2)/(2*side_a*side_b)
        radian_ab = np.arccos(cos_ab)
        angle_ab = (radian_ab*180)/math.pi

        if len(angle_ab_list) == 0:
            angle_ab_list.append(angle_ab)
        elif len(angle_ab_list) > 0:
            angle_ab_list.pop(0)
            angle_ab_list.append(angle_ab)

        # draw nose center
        cv2.circle(frame, ((L_LR_Cx_list[0] + R_LR_Cx_list[0])//2, (L_LR_Cy_list[0] + R_LR_Cy_list[0])//2), 5, (255, 255, 255), 5)

        # draw eye triangle
        cv2.line(frame, (L_LR_Cx_list[0], L_LR_Cy_list[0]), (R_LR_Cx_list[0], R_LR_Cy_list[0]), (255, 255, 0), 1)
        cv2.line(frame, (R_LR_Cx_list[0], L_LR_Cy_list[0]), (R_LR_Cx_list[0], R_LR_Cy_list[0]), (0, 255, 255), 1)
        cv2.line(frame, (L_LR_Cx_list[0], L_LR_Cy_list[0]), (R_LR_Cx_list[0], L_LR_Cy_list[0]), (255, 0, 255), 1)
        
        # draw eye rectangle
        cv2.rectangle(frame, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), (0, 255, 255), 2)
        index = index + 1

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - tick)
    cv2.putText(frame, "FPS:{} ".format(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2, cv2.LINE_AA)

    frame_count += 1
    print("[INFO] Frame End")

    # show frames local
    cv2.imshow("Hey, Wake Up!", frame)
    key = cv2.waitKey(1) & 0xFF

    # exit with "q"
    if key == ord("q"):
        break

# finish
cv2.destroyAllWindows()
vid.stop()
