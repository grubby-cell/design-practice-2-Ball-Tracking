"""
CALCULATION.PY

Side file for hosting the functions involved in
mathematical analysis of data.
"""
import numpy as np
from typing import List
from sklearn.metrics import r2_score


def percent_diff(expected, actual) -> float:
    """
    Calculates the percent difference between 2 values

    Args:
        expected: Expected value
        actual: Real/acquired value

    Returns:
        Float value
    """
    sign = 1 if expected > actual else -1
    value = (abs(actual - expected) / ((actual + expected) / 2)) * 100
    return sign * round(value, 2)


def slope(pos1: tuple, pos2: tuple) -> float:
    num = pos2[1] - pos1[1]
    den = pos2[0] - pos1[0]

    return round(num / den, 2)


def differentiate(data_set: List[tuple]) -> List[float]:
    diff_data = []
    for i in range(1, len(data_set) + 1):
        v_r = slope(data_set[i], data_set[i-1])
        diff_data.append(v_r)

    return diff_data


def polynomial_data(x, y, deg=2) -> dict:
    fit = np.polyfit(x, y, deg)
    polynomial = np.poly1d(fit)
    line = np.linspace(x[0], x[-1], max(y))
    poly_rel = round(r2_score(y, polynomial(x)), 4)
    coefficients = list(map(lambda c: float(c), fit))
    eq_comp = [
        f'{"+" if coefficients[i] > 0 else "-"} {abs(coefficients[i]):,.2f}t^{deg-i}'
        for i in range(deg + 1) if round(coefficients[i], 2) != 0
    ]
    poly_eq_form = ' '.join(eq_comp)

    return {
        'relation': poly_rel,
        'line': line,
        'polynomial': polynomial(line),
        'equation': poly_eq_form
    }

