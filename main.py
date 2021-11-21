"""
DESIGN PRACTICE 2: Fall 2021 Semester
Nagoya University, G30 Undergraduate Programs
Automotive Engineering
-----------------------
Choi, Franco, Giovanni, Nguyen, Nazirjonov
"""
from balltrack import BallTracker
from testfunctions import trial_scan
from videoscan import color_scan


def main():
    ballcam = BallTracker("ramp_trial.mp4")
    ballcam.track()
    # color_scan("ramp_trial.mp4")


if __name__ == "__main__":
    main()
