import numpy as np


class Event:
    def __init__(self, x, y, ts, p):
        self.x = x
        self.y = y
        self.ts = ts
        self.p = p

    def __str__(self):
        return'({}, {}, {}, {})'.format(self.x, self.y, self.ts, self.p)

    def __repr__(self):
        return '({}, {}, {}, {})'.format(self.x, self.y, self.ts, self.p)


class TimeSurface(object):
    def __init__(self, height, width, region_size, time_constant):
        self.height = height
        self.width = width
        self.r = region_size
        self.time_constant = time_constant

        self.latest_times = np.zeros((self.height, self.width))         # TODO: maybe this needs to be bigger so that we avoid border issues when time context of event at edge of array is calculated

        self.time_context = np.zeros((2 * self.r + 1, 2 * self.r + 1))

        self.time_surface = np.zeros_like(self.time_context)

    def _update_latest_times(self, event):
        """ create grid showing latest times at each spatial coordinate """

        self.latest_times[event.y, event.x] = event.ts

    def _update_time_context(self, event):
        """ create time context from grid of latest times """

        x, y = event.x, event.y
        self.time_context = self.latest_times[y - self.r:y + self.r + 1, x - self.r:x + self.r + 1]

    def _update_time_surface(self, current_time):
        """ create time surface from time context """

        self.time_surface = np.exp(-1. * (current_time - self.time_context) / self.time_constant)

    def process_event(self, event):
        """ update the latest times grid, time context and time surface uon receiving a new event """

        # ignore events at border

        if self.r < event.x < self.width - self.r and self.r < event.y < self.height - self.r:
            self._update_latest_times(event)
            self._update_time_context(event)
            self._update_time_surface(event.ts)

            return True
        else:
            return False
