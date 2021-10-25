import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
import datetime as dt


class BallTracker(object):
    def __init__(self, file_link: str):
        """
        Camera object initialization.

        Args:
            file_link (str): Link to video file

        Properties:
            video (cv2.VideoCapture): Video capture object
            is_running (bool): True if the camera is running
            color (tuple): RGB color of the rectangle
        """
        self.file = file_link
        self.video = cv2.VideoCapture(file_link, cv2.CAP_DSHOW)
        self.is_running = False
        self.color = (10, 255, 10)
        self.sensitivity = 25
        self.lower = np.array([0, 0, 255 - self.sensitivity])
        self.upper = np.array([255, self.sensitivity, 255])
        self.ball_points = []
        print("Ball tracker created!")

    def track_ball(self):
        self.is_running = True
        print(f'Opening file: {self.file}')

        while self.is_running:
            _, frame = self.video.read()

            if frame is None:
                break

            img = cv2.GaussianBlur(frame, (11, 11), 0)
            width, height = frame.shape[:2]

            mask = cv2.inRange(img, np.array([200, 170, 150]), np.array([220, 210, 200]))
            cnts_set = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts_set)
            center = None

            if len(cnts) > 0:
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                mts = cv2.moments(c)
                center = (int(mts["m10"] / mts["m00"]), int(mts["m01"] / mts["m00"]))

                # To see the centroid clearly
                if 5 < radius:
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 5)
                    cv2.imwrite("circled_frame.png", cv2.resize(frame, (int(height / 2), int(width / 2))))
                    cv2.circle(frame, center, 5, (0, 255, 0), -1)
                    print(round(radius, 3), center)
                    self.ball_points.append(center)

            cv2.imshow("Frame", mask)
            key = cv2.waitKey(10)
            if key == 27:
                print("Ending tracking session.")
                self.is_running = False
                break

        self.video.release()
        cv2.destroyAllWindows()

    def plot_data(self):
        x = [point[0] for point in self.ball_points]
        y = [point[1] for point in self.ball_points]
        plt.scatter(x, y)
        plt.xlim(0, max(x))
        plt.ylim(0, max(y))
        plt.xlabel("X-coordinates")
        plt.ylabel("Y-coordinates")
        plt.title('Ball motion')
        plt.show()

        print("-" * 25)
        print(f'Data points collected: {len(self.ball_points)}')