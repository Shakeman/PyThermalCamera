#!/usr/bin/env python3
"""Les Wright 21 June 2023.

https://youtube.com/leslaboratory

A Python program to read, parse and display thermal data from the Topdon TC001
Thermal camera!
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

print("Les Wright 21 June 2023")
print("https://youtube.com/leslaboratory")
print(
    "A Python program to read, parse and display thermal data from the Topdon TC001 Thermal camera!",
)
print()
print("Tested on Debian all features are working correctly")
print("This will work on the Pi However a number of workarounds are implemented!")
print("Seemingly there are bugs in the compiled version of cv2 that ships with the Pi!")
print()
print("Key Bindings:")
print()
print("a z: Increase/Decrease Blur")
print("s x: Floating High and Low Temp Label Threshold")
print(
    "d c: Change Interpolated scale Note: This will not change the window size on the Pi",
)
print("f v: Contrast")
print(
    "q w: Fullscreen Windowed (note going back to windowed does not seem to work on the Pi!)",
)
print("r t: Record and Stop")
print("p : Snapshot")
print("m : Cycle through ColorMaps")
print("h : Toggle HUD")


# We need to know if we are running on the Pi, because openCV behaves a little oddly on all the builds!
# https://raspberrypi.stackexchange.com/questions/5100/detect-that-a-python-program-is-running-on-the-pi
def is_raspberrypi() -> bool:
    """Check if the current device is a Raspberry Pi.

    This function attempts to determine if the current device is a Raspberry Pi
    by reading the device model information from the file located at
    "/sys/firmware/devicetree/base/model". If the file contains the string
    "raspberry pi" (case insensitive), the function returns True. Otherwise,
    it returns False.

    Returns:
        bool: True if the device is a Raspberry Pi, False otherwise.

    """
    try:
        with Path("/sys/firmware/devicetree/base/model").open() as m:
            if "raspberry pi" in m.read().lower():
                return True
    except Exception:  # noqa: BLE001, S110
        pass
    return False


is_pi: bool = is_raspberrypi()

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
# pull in the video but do NOT automatically convert to RGB, else it breaks the temperature data!
# https://stackoverflow.com/questions/63108721/opencv-setting-videocap-property-to-cap-prop-convert-rgb-generates-weird-boolean
cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)


# 256x192 General settings
width = 256  # Sensor width
height = 192  # sensor height
scale = 3  # scale multiplier
new_width = width * scale
new_height = height * scale
alpha = 1.0  # Contrast control (1.0-3.0)
colormap = 0
MAX_COLORMAPS = 11
font: int = cv2.FONT_HERSHEY_SIMPLEX
disp_fullscreen = False
cv2.namedWindow("Thermal", cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow("Thermal", new_width, new_height)
rad = 0  # blur radius
threshold = 2
hud = True
recording = False
elapsed = "00:00:00"
snaptime = "None"
start: float | None = None


def rec() -> cv2.VideoWriter:
    """Create and return a VideoWriter object for recording video."""
    now: str = time.strftime("%Y%m%d--%H%M%S")
    # do NOT use mp4 here, it is flakey!
    return cv2.VideoWriter(
        now + "output.avi",
        cv2.VideoWriter_fourcc(*"XVID"),
        25,
        (new_width, new_height),
    )


def snapshot(heatmap) -> str:  # noqa: ANN001, D103
    # I would put colons in here, but it Win throws a fit if you try and open them!
    now: str = time.strftime("%Y%m%d-%H%M%S")
    snaptime: str = time.strftime("%H:%M:%S")
    cv2.imwrite("TC001" + now + ".png", heatmap)
    return snaptime


while cap.isOpened():
    # Capture frame-by-frame
    ret: bool
    ret, frame = cap.read()
    if ret is True:
        imdata, thdata = np.array_split(frame, 2)
        # now parse the data from the bottom frame and convert to temp!
        # https://www.eevblog.com/forum/thermal-imaging/infiray-and-their-p2-pro-discussion/200/
        # Huge props to LeoDJ for figuring out how the data is stored and how to compute temp from it.
        # grab data from the center pixel...
        hi = thdata[96][128][0]
        lo = thdata[96][128][1]
        # print(hi,lo)
        lo = lo * 256
        rawtemp = hi + lo
        # print(rawtemp)
        temp = (rawtemp / 64) - 273.15
        temp = round(temp, 2)
        # print(temp)
        # break

        # find the max temperature in the frame
        lomax = thdata[..., 1].max()
        posmax = thdata[..., 1].argmax()
        # since argmax returns a linear index, convert back to row and col
        mcol, mrow = divmod(posmax, width)
        himax = thdata[mcol][mrow][0]
        lomax = lomax * 256
        maxtemp = himax + lomax
        maxtemp = (maxtemp / 64) - 273.15
        maxtemp = round(maxtemp, 2)

        # find the lowest temperature in the frame
        lomin = thdata[..., 1].min()
        posmin = thdata[..., 1].argmin()
        # since argmax returns a linear index, convert back to row and col
        lcol, lrow = divmod(posmin, width)
        himin = thdata[lcol][lrow][0]
        lomin = lomin * 256
        mintemp = himin + lomin
        mintemp = (mintemp / 64) - 273.15
        mintemp = round(mintemp, 2)

        # find the average temperature in the frame
        loavg = thdata[..., 1].mean()
        hiavg = thdata[..., 0].mean()
        loavg = loavg * 256
        avgtemp = loavg + hiavg
        avgtemp = (avgtemp / 64) - 273.15
        avgtemp = round(avgtemp, 2)

        # Convert the real image to RGB
        bgr = cv2.cvtColor(imdata, cv2.COLOR_YUV2BGR_YUYV)
        # Contrast
        bgr = cv2.convertScaleAbs(bgr, alpha=alpha)  # Contrast
        # bicubic interpolate, upscale and blur
        bgr = cv2.resize(
            bgr,
            (new_width, new_height),
            interpolation=cv2.INTER_CUBIC,
        )  # Scale up!
        if rad > 0:
            bgr = cv2.blur(bgr, (rad, rad))

        # apply colormap using case match
        match colormap:
            case 0:
                heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_JET)
                cmap_text = "Jet"
            case 1:
                heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_HOT)
                cmap_text = "Hot"
            case 2:
                heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_MAGMA)
                cmap_text = "Magma"
            case 3:
                heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_INFERNO)
                cmap_text = "Inferno"
            case 4:
                heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_PLASMA)
                cmap_text = "Plasma"
            case 5:
                heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_BONE)
                cmap_text = "Bone"
            case 6:
                heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_SPRING)
                cmap_text = "Spring"
            case 7:
                heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_AUTUMN)
                cmap_text = "Autumn"
            case 8:
                heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_VIRIDIS)
                cmap_text = "Viridis"
            case 9:
                heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_PARULA)
                cmap_text = "Parula"
            case 10:
                heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_RAINBOW)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                cmap_text = "Inv Rainbow"
            case _:
                heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_JET)
                cmap_text = "Jet"

        # print(heatmap.shape)

        # draw crosshairs
        cv2.line(
            heatmap,
            (int(new_width / 2), int(new_height / 2) + 20),
            (int(new_width / 2), int(new_height / 2) - 20),
            (255, 255, 255),
            2,
        )  # vline
        cv2.line(
            heatmap,
            (int(new_width / 2) + 20, int(new_height / 2)),
            (int(new_width / 2) - 20, int(new_height / 2)),
            (255, 255, 255),
            2,
        )  # hline

        cv2.line(
            heatmap,
            (int(new_width / 2), int(new_height / 2) + 20),
            (int(new_width / 2), int(new_height / 2) - 20),
            (0, 0, 0),
            1,
        )  # vline
        cv2.line(
            heatmap,
            (int(new_width / 2) + 20, int(new_height / 2)),
            (int(new_width / 2) - 20, int(new_height / 2)),
            (0, 0, 0),
            1,
        )  # hline
        # show temp
        cv2.putText(
            heatmap,
            str(temp) + " C",
            (int(new_width / 2) + 10, int(new_height / 2) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            heatmap,
            str(temp) + " C",
            (int(new_width / 2) + 10, int(new_height / 2) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if hud is True:
            # display black box for our data
            cv2.rectangle(heatmap, (0, 0), (160, 120), (0, 0, 0), -1)
            # put text in the box
            cv2.putText(
                heatmap,
                "Avg Temp: " + str(avgtemp) + " C",
                (10, 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                heatmap,
                "Label Threshold: " + str(threshold) + " C",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                heatmap,
                "Colormap: " + cmap_text,
                (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                heatmap,
                "Blur: " + str(rad) + " ",
                (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                heatmap,
                "Scaling: " + str(scale) + " ",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                heatmap,
                "Contrast: " + str(alpha) + " ",
                (10, 84),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                heatmap,
                "Snapshot: " + snaptime + " ",
                (10, 98),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            if recording is False:
                cv2.putText(
                    heatmap,
                    "Recording: " + elapsed,
                    (10, 112),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )
            if recording is True:
                cv2.putText(
                    heatmap,
                    "Recording: " + elapsed,
                    (10, 112),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (40, 40, 255),
                    1,
                    cv2.LINE_AA,
                )

        # Yeah, this looks like we can probably do this next bit more efficiently!
        # display floating max temp
        if maxtemp > avgtemp + threshold:
            cv2.circle(heatmap, (mrow * scale, mcol * scale), 5, (0, 0, 0), 2)
            cv2.circle(heatmap, (mrow * scale, mcol * scale), 5, (0, 0, 255), -1)
            cv2.putText(
                heatmap,
                str(maxtemp) + " C",
                ((mrow * scale) + 10, (mcol * scale) + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                heatmap,
                str(maxtemp) + " C",
                ((mrow * scale) + 10, (mcol * scale) + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # display floating min temp
        if mintemp < avgtemp - threshold:
            cv2.circle(heatmap, (lrow * scale, lcol * scale), 5, (0, 0, 0), 2)
            cv2.circle(heatmap, (lrow * scale, lcol * scale), 5, (255, 0, 0), -1)
            cv2.putText(
                heatmap,
                str(mintemp) + " C",
                ((lrow * scale) + 10, (lcol * scale) + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                heatmap,
                str(mintemp) + " C",
                ((lrow * scale) + 10, (lcol * scale) + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # display image
        cv2.imshow("Thermal", heatmap)

        key_press: int = cv2.waitKey(1)
        if key_press == ord("a"):  # Increase blur radius
            rad += 1
        if key_press == ord("z"):  # Decrease blur radius
            rad -= 1
            rad = max(0, rad)

        if key_press == ord("s"):  # Increase threshold
            threshold += 1
        if key_press == ord("x"):  # Decrease threashold
            threshold -= 1
            threshold = max(0, threshold)

        if key_press == ord("d"):  # Increase scale
            scale += 1
            scale = min(5, scale)
            new_width: int = width * scale
            new_height: int = height * scale
            if disp_fullscreen is False and is_pi is False:
                cv2.resizeWindow("Thermal", new_width, new_height)
        if key_press == ord("c"):  # Decrease scale
            scale -= 1
            scale = max(1, scale)
            new_width = width * scale
            new_height = height * scale
            if disp_fullscreen is False and is_pi is False:
                cv2.resizeWindow("Thermal", new_width, new_height)

        if key_press == ord("q"):  # enable fullscreen
            disp_fullscreen = True
            cv2.namedWindow("Thermal", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(
                "Thermal",
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )
        if key_press == ord("w"):  # disable fullscreen
            disp_fullscreen = False
            cv2.namedWindow("Thermal", cv2.WINDOW_GUI_NORMAL)
            cv2.setWindowProperty(
                "Thermal",
                cv2.WND_PROP_AUTOSIZE,
                cv2.WINDOW_GUI_NORMAL,
            )
            cv2.resizeWindow("Thermal", new_width, new_height)

        if key_press == ord("f"):  # contrast+
            alpha += 0.1
            alpha = round(alpha, 1)  # fix round error
            alpha = min(3.0, alpha)
        if key_press == ord("v"):  # contrast-
            alpha -= 0.1
            alpha: float = round(alpha, 1)  # fix round error
            if alpha <= 0:
                alpha = 0.0

        if key_press == ord("h"):
            if hud is True:
                hud = False
            elif hud is False:
                hud = True

        if key_press == ord("m"):  # m to cycle through color maps
            colormap = 0 if colormap == MAX_COLORMAPS else colormap + 1

        if key_press == ord("r") and recording is False:  # r to start reording
            video_out: cv2.VideoWriter = rec()
            recording = True
            start = time.time()
        if key_press == ord("t"):  # f to finish reording
            recording = False
            elapsed = "00:00:00"

        if key_press == ord("p"):  # f to finish reording
            snaptime: str = snapshot(heatmap)

        if key_press == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            break

        if recording is True and start is not None:
            elapsed: str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
            # print(elapsed)
            video_out.write(heatmap)
