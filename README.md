# Hey Wake Up

<br />

<div align="center">

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://github.com/DEVHEE/Hey-Wake-Up)
[![PyPI status](https://img.shields.io/pypi/status/ansicolortags.svg)]()
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![made-with-opencv](http://img.shields.io/badge/OpenCV-5c3ee8?style=square&logo=OpenCV&logoColor=white)]()
[![Raspberry Pi](http://img.shields.io/badge/Raspberry%20Pi-c51a4a?style=square&logo=Raspberry-Pi&logoColor=white)]()

[![made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://GitHub.com/DEVHEE/)
[![cc-nc-sa](http://ForTheBadge.com/images/badges/cc-nc-sa.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0)

</div>

> This system is currently under development and is always being updated.

## Table of contents

-   [Folder Structure](#folder-structure)
-   [Raspberry Pi](#raspberry-pi)
-   [Detector and Predictor](#detector-and-predictor)
    -   [OpenCV](#opencv)
    -   [Cascade Classifier](#cascade-classifier)
        -   [Frontal Face Detection](#frontal-face-detection)
        -   [Eye Detection](#eye-detection)
    -   [Dlib](#dlib)
        -   [68 Face Landmarks](#68-face-landmarks)
        -   [Drowsiness Detection](#drowsiness-detection)
        -   [Live Rotation](#live-rotation)
    -   [Gaze Tracking](#gaze-tracking)
-   [Flask](#flask)
    -   [Live Streaming](#live-streaming)

## Folder Structure

    .
    ├── LICENSE
    ├── README.md
    ├── app.py
    ├── data
    │   ├── etc
    │   │   └── NanumGothic-Bold.ttf
    │   ├── library
    │   │   └── gaze_tracking
    │   │       ├── __init__.py
    │   │       ├── calibration.py
    │   │       ├── eye.py
    │   │       ├── gaze_tracking.py
    │   │       └── pupil.py
    │   └── model
    │       ├── cascade
    │       │   ├── haarcascade_eye.xml
    │       │   ├── haarcascade_frontalface_alt.xml
    │       │   ├── haarcascade_frontalface_alt2.xml
    │       │   ├── haarcascade_frontalface_alt_tree.xml
    │       │   ├── haarcascade_frontalface_default.xml
    │       │   └── lbpcascade_frontalface.xml
    │       └── landmark
    │           └── shape_predictor_68_face_landmarks.dat
    ├── main.py
    ├── temp
    │   ├── fps_optimize.py
    │   └── temp.py
    └── templates
        └── index.html

## Licensing

The GNU GPLv3.0 License 2021 KIM DONGHEE
