"""
GENERAL.PY

Side file for hosting the general-purpose functions
and decorators. The Board class holds the data for
the physical properties of the real board.
"""
import time


class Board(object):
    """
    Store constants for project board.
    """
    WIDTH = 78
    LENGTH = 117
    r_width = 0
    r_length = 0


def timer(func):
    def timer_wrapper(*args, **kwargs):
        start_time = time.time()
        ret_val = func(*args, **kwargs)
        runtime = time_interval(start_time)
        print(f'\n[Elapsed time: {runtime}]')
        return ret_val

    return timer_wrapper


def time_interval(start_period: float) -> str:
    """
    Calculates time interval, given a starting time.

    :param start_period: Time of start
    :return:
    """
    def convert_time_diff(duration):
        duration = round(duration, 2)
        d_hours = int(duration // 3600)
        d_minutes = int((duration - (d_hours * 3600)) // 60)
        d_seconds = int(round(duration - (d_hours * 3600) - (d_minutes * 60)))

        return d_hours, d_minutes, d_seconds

    try:
        now = time.time()
        hours, minutes, seconds = convert_time_diff(now - start_period)
        time_elapsed = []

        if hours:
            time_elapsed.append(f'{hours}h')
        if minutes:
            time_elapsed.append(f'{minutes}m')
        if seconds:
            time_elapsed.append(f'{seconds}s')

        return str(' '.join(time_elapsed))

    except Exception as e:
        print(f'Time interval computation error: {e}')
        return "ERROR"
