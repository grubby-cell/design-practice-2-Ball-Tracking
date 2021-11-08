import time

BOARD_WIDTH = 78
BOARD_LENGTH = 117


def timer(func):
    def timer_wrapper(*args, **kwargs):
        start_time = time.time()
        ret_val = func(*args, **kwargs)
        runtime = time.time() - start_time
        print(f'\n[Elapsed time: {round(runtime, 2)}s]')
        return ret_val

    return timer_wrapper


def slope(pos1: tuple, pos2: tuple) -> float:
    num = pos2[1] - pos1[1]
    den = pos2[0] - pos1[0]

    return round(num/den, 2)


def differentiate(data_set: list) -> list:
    diff_data = []
    for i in range(1, len(data_set)+1):
        v_r = slope(data_set[i], data_set[i-1])
        diff_data.append(v_r)

    return diff_data
