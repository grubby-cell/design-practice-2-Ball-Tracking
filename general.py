import time
import math


class Board(object):
    WIDTH = 78
    LENGTH = 117


def timer(func):
    def timer_wrapper(*args, **kwargs):
        start_time = time.time()
        ret_val = func(*args, **kwargs)
        runtime = time.time() - start_time
        print(f'\n[Elapsed time: {round(runtime, 2)}s]')
        return ret_val

    return timer_wrapper


def time_interval(start_period: float):
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
        hours, minutes, seconds = convert_time_diff(time.time() - start_period)
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


def slope(pos1: tuple, pos2: tuple) -> float:
    num = pos2[1] - pos1[1]
    den = pos2[0] - pos1[0]

    return round(num / den, 2)


def differentiate(data_set: list) -> list:
    diff_data = []
    for i in range(1, len(data_set) + 1):
        v_r = slope(data_set[i], data_set[i - 1])
        diff_data.append(v_r)

    return diff_data
