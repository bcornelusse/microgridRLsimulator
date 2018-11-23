from datetime import timedelta


def datetime_range(start, end, delta):
    current = start
    if not isinstance(delta, timedelta):
        delta = timedelta(**delta)
    while current < end:
        yield current
        current += delta


TOL_IS_ZERO = 2.5 * 2e-2


def negative(value, tol=TOL_IS_ZERO):
    """
    Check if a value is negative with respect to a tolerance.
    :param value: Value
    :param tol: Tolerance.
    :return: Boolean.
    """
    return value < -tol


def positive(value, tol=TOL_IS_ZERO):
    """
    Check if a value is positive with respect to a tolerance.
    :param value: Value
    :param tol: Tolerance.
    :return: Boolean.
    """
    return value > tol
