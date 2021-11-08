import cv2
import time
import numpy as np
import datetime as dt
from imutils import resize
from general import Board, timer, time_interval


class BallTracker(object):
    def __init__(self, file_name: str):
        """
        Camera object initialization.
        """
        self.video = None
        self.file = file_name
        self.is_running = False
        self.board = Board()
        self.output = None

        # Video properties
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.color = (10, 255, 10)
        self.lower = np.array([52, 35, 0])
        self.upper = np.array([255, 255, 255])
        self.ball = []
        self.ball_detected = False
        print("Ball tracker created!")

    @timer
    def track(self):
        self.video = cv2.VideoCapture(self.file)
        print("Beginning trial scan.")
        start_time = time.time()
        center = None
        radius = 0
        pos = {'x': 0, 'y': 0}
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        while self.video.isOpened():
            ret, frame = self.video.read()

            if not ret:
                break

            # Setting up frame and region-of-interest
            width, height, _ = frame.shape
            roi = frame[70:width - 50, 350:height - 200]
            roi = resize(roi, height=540)
            r_height, r_width, _ = roi.shape
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lower_hue = np.array([52, 35, 0])
            upper_hue = np.array([255, 255, 255])
            mask = cv2.inRange(hsv, lower_hue, upper_hue)
            (contours, _) = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                c = max(contours, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)

                try:
                    if radius > 10:
                        self.ball_detected = True
                        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                        pos['x'], pos['y'] = round(x), round(y)
                        cv2.circle(roi, center, 10, (30, 200, 255), -1)
                        if len(self.ball) > 30:
                            del self.ball[0]

                        self.ball.append(center)

                    else:
                        self.ball_detected = False

                except Exception as e:
                    print(f'Error during trace: {e}')

                if len(self.ball) > 2:
                    for i in range(1, len(self.ball)):
                        dx = abs(self.ball[i-1][0] - self.ball[i][0])
                        dy = abs(self.ball[i - 1][1] - self.ball[i][1])
                        if dx < 100 and dy < 50:
                            cv2.line(roi, self.ball[i - 1], self.ball[i], (10, 10, 255), 5)

            time_stamp = dt.datetime.now().strftime("%I:%M:%S %p")
            frame_text = [
                time_stamp,
                f'Runtime: {time_interval(start_time)}',
                f'Radius: {round(radius, 2)}',
                f'Ball in frame: {"YES" if self.ball_detected else "NO"}',
                f'Last position: ({pos["x"]},{pos["y"]})'
            ]
            for i, label in enumerate(frame_text):
                cv2.putText(roi, str(label), (10, 20 + 20 * i), self.font, 0.5, (10, 255, 100), 1)

            cv2.imshow("Ball Tracking", roi)

            key = cv2.waitKey(10)
            if key == 27:
                print("Ending tracking session.")
                break

        self.video.release()
        cv2.destroyAllWindows()
