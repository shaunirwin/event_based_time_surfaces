import numpy as np

from lib.spatio_temporal_feature import TimeSurface, Event


def test_time_surface_init():
    ts = TimeSurface(height=30, width=40, region_size=2, time_constant=1000)

    assert ts.height == 30
    assert ts.width == 40
    assert ts.r == 2
    assert ts.time_constant == 1000

    assert ts.latest_times.shape == (30, 40)
    assert ts.time_context.shape == (5, 5)
    assert ts.time_surface.shape == (5, 5)


def test_time_surface_init_different_values():
    ts = TimeSurface(height=30, width=35, region_size=4, time_constant=400)

    assert ts.height == 30
    assert ts.width == 35
    assert ts.r == 4
    assert ts.time_constant == 400

    assert ts.latest_times.shape == (30, 35)
    assert ts.time_context.shape == (9, 9)
    assert ts.time_surface.shape == (9, 9)


def test_process_event():
    ts = TimeSurface(height=30, width=40, region_size=2, time_constant=1000)

    ts.process_event(Event(23, 12, 340., 1))

    assert ts.latest_times[12, 23] == 340

    assert ts.time_context.shape == (5, 5)
    assert ts.time_surface.shape == (5, 5)

    assert ts.time_context[0, 0] == 0
    assert ts.time_context[1, 1] == 0
    assert ts.time_context[2, 2] == 340
    assert ts.time_context[3, 3] == 0
    assert ts.time_context[4, 4] == 0

    assert ts.time_surface[0, 0] == np.exp(-(340-0)/1000.)
    assert ts.time_surface[1, 1] == np.exp(-(340-0)/1000.)
    assert ts.time_surface[2, 2] == np.exp(-(340-340)/1000.)
    assert ts.time_surface[3, 3] == np.exp(-(340-0)/1000.)
    assert ts.time_surface[4, 4] == np.exp(-(340-0)/1000.)


def test_ignore_events_at_border():
    ts = TimeSurface(height=30, width=40, region_size=2, time_constant=1000)

    ts.process_event(Event(0, 12, 340., 1))

    assert ts.latest_times[12, 0] == 0

    assert ts.time_context.shape == (5, 5)
    assert ts.time_surface.shape == (5, 5)

    assert ts.time_context[0, 0] == 0
    assert ts.time_context[1, 1] == 0
    assert ts.time_context[2, 2] == 0
    assert ts.time_context[3, 3] == 0
    assert ts.time_context[4, 4] == 0

    assert ts.time_surface[0, 0] == 0
    assert ts.time_surface[1, 1] == 0
    assert ts.time_surface[2, 2] == 0
    assert ts.time_surface[3, 3] == 0
    assert ts.time_surface[4, 4] == 0
