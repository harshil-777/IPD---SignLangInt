"""Webcam capture helper.

On Windows, OpenCV's default camera backend (MSMF) can hang indefinitely
on `read()` for some camera drivers, with no error and no window ever
appearing. DirectShow (CAP_DSHOW) is the standard workaround and is tried
first there.
"""
from __future__ import annotations

import sys

import cv2


def open_camera(index: int = 0) -> cv2.VideoCapture:
    if sys.platform == "win32":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()

    return cv2.VideoCapture(index)
