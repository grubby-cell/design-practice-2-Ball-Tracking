"""
DESIGN PRACTICE 2: Fall 2021 Semester
Nagoya University, G30 Undergraduate Programs
Automotive Engineering
-----------------------
Choi, Franco, Giovanni, Nguyen, Nazirjonov
"""
from balltrack import BallTracker


def main():
    FILENAME = "trial2-1.mp4"
    ballcam = BallTracker(FILENAME)
    ballcam.track()


if __name__ == "__main__":
    main()
