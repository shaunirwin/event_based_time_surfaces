import numpy as np


class TimeSurface(object):
    def __init__(self, height, width, region_size, time_constant):
        self.height = height
        self.width = width

        self.latest_times_on = np.zeros((self.height, self.width))
        self.latest_times_off = np.zeros_like(self.latest_times_on)

        self.time_context_on = np.zeros_like(self.latest_times_on)
        self.time_context_off = np.zeros_like(self.latest_times_on)

        self.time_surface_on = np.zeros_like(self.latest_times_on)
        self.time_surface_off = np.zeros_like(self.latest_times_on)

        self.r = region_size
        self.time_constant = time_constant

    def _update_latest_times(self, event):
        """ create grid showing latest times at each spatial coordinate """

        if event.p:
            self.latest_times_on[event.y, event.x] = event.ts
        else:
            self.latest_times_off[event.y, event.x] = event.ts

    def _update_time_context(self, event):
        """ create time context from grid of latest times """

        for x in range(event.x - self.r, event.x + self.r):
            for y in range(event.y - self.r, event.y + self.r):
                if self.r <= x <= self.width - self.r and self.r <= y <= self.height - self.r:
                    if event.p:
                        self.time_context_on[y, x] = \
                            np.max(self.latest_times_on[y - self.r:y + self.r, x - self.r:x + self.r])
                    else:
                        self.time_context_off[y, x] = \
                            np.max(self.latest_times_off[y - self.r:y + self.r, x - self.r:x + self.r])

    def _update_time_surface(self, event_on, current_time):
        """ create time surface from time context """

        if event_on:
            self.time_surface_on = np.exp(-(current_time - self.time_context_on) / self.time_constant)
        else:
            self.time_surface_off = np.exp(-(current_time - self.time_context_off) / self.time_constant)

    def process_event(self, event):
        """ update the latest times grid, time context and time surface uon receiving a new event """

        self._update_latest_times(event)
        self._update_time_context(event)
        self._update_time_surface(event.p, event.ts)
