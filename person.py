from collections import deque
import numpy as np

class Person:

    trajectory = None
    id = None
    check = False

    def __init__(self, buffer_size=10):
        self.trajectory = deque(maxlen=buffer_size)

    def set_id(self, id):
        self.id = id

    def set_check(self):
        self.check = True

    def reset_check(self):
        self.check = False

    def add_location(self, position):
        self.trajectory.append(position)

    def get_id(self):
        return self.id

    def get_check(self):
        return self.check

    def get_last_position(self):
        return self.trajectory[-1]

    def get_trajectory(self):
        return self.trajectory

    def get_center_trajectory(self):
        center_trajectory = []
        trajectory = self.trajectory
        for x1, y1, x2, y2 in trajectory:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            center_trajectory.append((cx, cy))

        return center_trajectory
