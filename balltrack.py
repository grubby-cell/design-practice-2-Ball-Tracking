"""
BALLTRACK.PY

Main ball-tracking class and tools for the project. Uses
OpenCV for video capture and processing, then NumPy and
custom module for post-processing and computations. Data
is plotted visually using Matplotlib.
"""
import cv2
import time
import csv
import logging as lg
import numpy as np
import datetime as dt
import traceback as tb
from os import path
from imutils import resize
from time import perf_counter
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from general import timer, time_interval
from datamodels import Board, Point
from calculation import polynomial_data


class BallTracker(object):
    def __init__(self, file_name: str):
        """
        Camera object initialization.
        """
        # Basic run properties
        self.video = None
        self.file = file_name
        self.is_running = False
        self.board = Board()
        self.output = None

        # Video properties
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.lower = np.array([52, 35, 0])
        self.upper = np.array([255, 255, 255])
        self.ball = []
        self.radius_log = []
        self.ball_detected = False
        self.frame_dim = {'length': 0, 'width': 0}
        self.roi_dim = {'length': 0, 'width': 0}

        # Configure logging
        log_fmt = "[%(levelname)s] %(asctime)s | %(message)s"
        date_fmt = "%I:%M:%S %p"
        lg.basicConfig(
            level=lg.DEBUG,
            format=log_fmt,
            datefmt=date_fmt
        )
        lg.getLogger('matplotlib.font_manager').setLevel(lg.WARNING)

        # Notification of completion
        lg.info("Ball tracker created.")
        print(f'Board dimensions: {self.board.WIDTH}mm x {self.board.LENGTH}mm')
        print("-"*25)

    def __repr__(self):
        return f'<BallTracker file={self.file}, bounds={self.lower}, {self.upper}>'

    def __len__(self):
        if self.video is not None:
            return int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        return 0

    def __getitem__(self, item: int):
        if type(item) != int:
            type_name = lambda v: str(type(v)).split("'")[1].lower()
            raise ValueError(f'{type_name(item)} was given but int was required.')
        return self.ball[item]

    def __iter__(self):
        for item in self.ball:
            yield item

    def trace_path(self, frame):
        """
        Traces path line of ball in video.
        """
        for i in range(1, len(self.ball)):
            prev = self.ball[i-1]
            curr = self.ball[i]
            dx = abs(prev.x - curr.x)
            dy = abs(prev.y - curr.y)
            if dx < 100 and dy < 50:
                p_prev = (prev.x, prev.y)
                p_curr = (curr.x, curr.y)
                cv2.line(frame, p_prev, p_curr, (10, 10, 255), 5)

    @timer
    def track(self):
        """
        Track ball and plot position markers on frame.
        """
        # Configure starting variables and preprocessing
        self.video = cv2.VideoCapture(self.file)
        print("Beginning tracking sequence.")
        start_time = time.time()
        radius = 0
        pos = {'x': 0, 'y': 0}
        baseline = perf_counter()

        while self.video.isOpened():
            # Grab frame from video
            ret, frame = self.video.read()
            if not ret:
                break

            # Setting up frame and region-of-interest
            width, height, _ = frame.shape
            self.roi_dim = {
                'length': (350, height-200),
                'width': (70, width-50)
            }
            self.frame_dim = {
                'length': (0, height),
                'width': (0, width)
            }
            roi = frame[
                  self.roi_dim['width'][0]:self.roi_dim['width'][1],
                  self.roi_dim['length'][0]:self.roi_dim['length'][1]
            ]
            frame = resize(frame, height=540)
            roi = resize(roi, height=540)
            r_height, r_width, _ = roi.shape
            self.board.r_width, self.board.r_length = r_height, r_width

            # Isolate ball shape via HSV bounds
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lower_hue = np.array([52, 35, 0])
            upper_hue = np.array([255, 255, 255])
            mask = cv2.inRange(hsv, lower_hue, upper_hue)
            (contours, _) = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # When "ball" is detected
            if len(contours) > 0:
                c = max(contours, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                mts = cv2.moments(c)

                # Highlight ball shape and collect position data
                try:
                    if 10 < radius < 16 and x < 700:
                        self.ball_detected = True
                        stamp = time.perf_counter()
                        center = (int(mts["m10"]/mts["m00"]), int(mts["m01"]/mts["m00"]))
                        pos['x'], pos['y'] = round(x), round(y)
                        frame_data = Point(
                            x=center[0],
                            y=center[1],
                            time=stamp
                        )
                        self.ball.append(frame_data)
                        self.radius_log.append(round(radius, 2))
                        cv2.circle(roi, center, 15, (30, 200, 255), 3)

                    else:
                        self.ball_detected = False

                    # Put tracer line to show ball path
                    if len(self.ball) > 2:
                        self.trace_path(roi)

                except Exception as e:
                    lg.error(f'Error during trace: {e}')
                    tb.print_exc()

            # Put window text and frame details
            cv2.putText(frame, "Video feed", (10, 30), self.font, 0.7, (10, 255, 100), 2)
            time_stamp = dt.datetime.now().strftime("%I:%M:%S %p")
            roi_text = [
                time_stamp,
                f'Runtime: {time_interval(start_time)}',
                f'Radius: {round(radius, 2)}',
                f'Ball in frame: {"Yes" if self.ball_detected else "No"}',
                f'Last position: ({pos["x"]},{pos["y"]})'
            ]
            for i, label in enumerate(roi_text):
                cv2.putText(roi, str(label), (10, 20 + 20 * i),
                            self.font, 0.5, (10, 255, 100), 1)

            # cv2.imshow("Original Clip", frame)
            cv2.imshow("Ball Tracking", roi)

            # Close windows with 'Esc' key
            key = cv2.waitKey(10)
            if key == 27:
                print("Ending tracking session.")
                break

        self.video.release()
        cv2.destroyAllWindows()
        lg.info("Tracker session ended.")

        # Postprocessing functions for data analysis
        self.get_run_statistics()
        # self.export_data()

    def get_run_statistics(self):
        """
        Summarize data gathered during tracking run.
        """
        avg = lambda d: round(sum(d) / len(d), 2)
        print("\n***** TRACK DATA *****")
        print(f'Radius range: [{min(self.radius_log)}, {max(self.radius_log)}]')
        print(f'Average radius: {avg(self.radius_log)}')
        print(f'Total points: {len(self.ball)}')
        print(f'ROI area: W = {self.roi_dim["width"]}, L = {self.roi_dim["length"]}')
        print("*" * 22)

        self.plot_data()

    def plot_data(self):
        """
        Plot 2D data acquired from tracking session.
        """
        # Separate coordinates data into X and Y parameters
        x_raw = [p.x for p in self.ball]
        y_raw = [self.roi_dim['width'][1] - p.y for p in self.ball]
        x_data = np.array(x_raw, dtype=float)
        y_data = np.array(y_raw, dtype=float)
        bestfit = polynomial_data(x_raw, y_raw, 3)
        print("-" * 25)
        print(f"y(x) = {bestfit['equation']}")
        print(f"Polynomial R-squared: {bestfit['relation']}")

        # Generate plot
        plt.scatter(x_data, y_data, color="firebrick")
        plt.plot(bestfit['line'], bestfit['polynomial'], color="gold")
        plt.title("Ball point tracking")
        plt.xlim(0, np.max(x_data)+10)
        plt.ylim(np.min(y_data)-10, np.max(y_data)+10)
        plt.xlabel("X-coordinate")
        plt.ylabel("Y-coordinate")
        plt.grid()
        plt.show()

    def export_data(self):
        """
        Export gathered data as CSV file.
        """
        plain_file_name = self.file.split(".")[0]
        csv_file = f'data_({plain_file_name}).csv'

        if not path.exists(csv_file):
            with open(csv_file, 'w', encoding='UTF8') as f:
                header = ["Time", "X", "Y"]
                writer = csv.writer(f)
                writer.writerow(header)

                # Go through list of stored points, write to file
                for point in self.ball:
                    data_input = [
                        str(point.time),
                        str(point.x),
                        str(point.y)
                    ]
                    writer.writerow(data_input)

            print("Data compiled and exported to CSV file.")

        else:
            print(f'Data for \"{self.file}\" has already been exported to CSV file.')
