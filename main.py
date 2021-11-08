from balltrack import BallTracker
from testing import trial_scan
from videoscan import scan_video

if __name__ == "__main__":
    ballcam = BallTracker("ramp_trial.mp4")
    ballcam.track()
    # scan_video("ramp_trial.mp4")
