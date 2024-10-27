"""Module to capture video from a thermal camera and display it using OpenCV."""


import argparse

import cv2

parser = argparse.ArgumentParser()
parser.add_argument(
    "--device",
    type=int,
    default=0,
    help="Video Device number e.g. 0, use v4l2-ctl --list-devices",
)
args: argparse.Namespace = parser.parse_args()

dev = args.device if args.device else 0


# init video
cap = cv2.VideoCapture("/dev/video" + str(dev), cv2.CAP_V4L)
# cap = cv2.VideoCapture(0)

# we need to set the resolution here why?
"""
wright@CF-31:~/Desktop$ v4l2-ctl --list-formats-ext
ioctl: VIDIOC_ENUM_FMT
	Index       : 0
	Type        : Video Capture
	Pixel Format: 'YUYV'
	Name        : YUYV 4:2:2
		Size: Discrete 256x192
			Interval: Discrete 0.040s (25.000 fps)
		Size: Discrete 256x384
			Interval: Discrete 0.040s (25.000 fps)
"""

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cv2.namedWindow("Thermal", cv2.WINDOW_GUI_NORMAL)
font: int = cv2.FONT_HERSHEY_SIMPLEX

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret is True:
        cv2.namedWindow("Thermal", cv2.WINDOW_NORMAL)
        cv2.imshow("Thermal", frame)

        key_press: int = cv2.waitKey(3)
        if key_press == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            break
