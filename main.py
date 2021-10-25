"""
MAIN.PY

File for main project run configuration.
"""
from balltrack import BallTracker

if __name__ == "__main__":
    FILENAME = "trial-edit.mp4"
    cap = BallTracker(FILENAME)
    cap.track_ball()