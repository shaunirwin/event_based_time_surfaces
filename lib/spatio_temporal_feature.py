import numpy as np


class TimeSurface(object):
    def __init__(self, height, width, region_size, time_constant):
        self.height = height
        self.width = width

        self.latest_times = np.zeros((self.height, self.width))

        self.time_context = np.zeros_like(self.latest_times)

        self.time_surface = np.zeros_like(self.latest_times)

        self.r = region_size
        self.time_constant = time_constant

    def _update_latest_times(self, event):
        """ create grid showing latest times at each spatial coordinate """

        self.latest_times[event.y, event.x] = event.ts

    def _update_time_context(self, event):
        """ create time context from grid of latest times """

        for x in range(event.x - self.r, event.x + self.r):
            for y in range(event.y - self.r, event.y + self.r):
                if self.r <= x <= self.width - self.r and self.r <= y <= self.height - self.r:
                    self.time_context[y, x] = \
                            np.max(self.latest_times[y - self.r:y + self.r, x - self.r:x + self.r])

    def _update_time_surface(self, current_time):
        """ create time surface from time context """

        self.time_surface = np.exp(-(current_time - self.time_context) / self.time_constant)

    def process_event(self, event):
        """ update the latest times grid, time context and time surface uon receiving a new event """

        self._update_latest_times(event)
        self._update_time_context(event)
        self._update_time_surface(event.ts)
