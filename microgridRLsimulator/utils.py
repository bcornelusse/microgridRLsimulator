from datetime import timedelta, datetime


def datetime_range(start, end, delta):
    current = start
    if not isinstance(delta, timedelta):
        delta = timedelta(**delta)
    while current < end:
        yield current
        current += delta


# TOL_IS_ZERO = 2.5 * 2e-2
TOL_IS_ZERO = 1e-5  # A higher tolerance leads to large difference between the optimization problem and simulation


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


def decode_GridState(gridstates, features, n_sequences):
    """

    :param features: A features dict
    :param n_sequences: the number of state sequences (backcast)
    :return a list of the state values
    """
    values = list()
    for gridstate in gridstates:
        for attr, val in sorted(features.items()):
            if val:
                x = getattr(gridstate, attr)
                if isinstance(x, list):
                    values += x
                else:
                    values.append(x)
        if gridstate == gridstates[0]:
            state_alone_size = len(values)
    n_missing_values = state_alone_size * (n_sequences - len(gridstates))
    values = n_missing_values * [.0] + values
    return values


def time_string_for_storing_results(name, case):
    """

    :param case: the case name
    :return a string used for file or folder names
    """
    return name + "_%s_%s" % (case, datetime.now().strftime('%Y-%m-%d_%H%M'))
