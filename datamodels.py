import math
from pydantic import BaseModel, validator


class Board(object):
    """
    Store constants for project board.
    """
    WIDTH = 78
    LENGTH = 117
    r_width = 0
    r_length = 0


class Point(BaseModel):
    """
    Data class representing each recorded data point taken
    from the video capture. Stores coordinates and timestamp.
    """
    x: int
    y: int
    time: float

    def __repr__(self):
        return f'<Point position=({self.x},{self.y}) time={self.time:.3f}>'

    def __len__(self):
        return math.sqrt(self.x**2 + self.y**2)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.time == other.time

    @validator("x")
    def check_x_value(cls, x):
        if x < 0 or x > 1500:
            raise ValueError("X-coordinate is out of bounds.")

        return x

    @validator("y")
    def check_y_value(cls, y):
        if y < 0 or y > 1500:
            raise ValueError("Y-coordinate is out of bounds.")
        return y

    @validator("time")
    def is_time_valid(cls, t):
        if t < 0:
            raise ValueError("Invalid time given.")
        return t
