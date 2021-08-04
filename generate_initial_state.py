import numpy as np


def get_pyramid(l, b, n_secs):
    """"Create pyramid-like shape with fixed length of (upper) sides discretized in equal length elements.

    Args:
        l (float): Total length of sides.
        b (float): Length of base.
        n_secs (int): Number of equal length elements to discretize the sides in.

    Returns:
        tuple: x,y-coordinates of discretized pyramid - excluding point at origin.

    """
    if n_secs % 2 == 0:
        sag = (l ** 2 - b ** 2) ** .5
        x = np.linspace(0, b, n_secs + 1)[1:]
        y = np.hstack((np.linspace(0, -sag/2, n_secs//2+1)[1:], np.linspace(-sag/2, 0, n_secs//2+1)[1:]))
    else:
        l_sec = l/n_secs
        sag = ((l-l_sec) ** 2 - (b - l_sec) ** 2) ** .5
        x = np.hstack((np.linspace(0, (b - l_sec) / 2, n_secs // 2 + 1)[1:],
                       np.linspace((b - l_sec) / 2 + l_sec, b, n_secs // 2 + 1)))
        y = np.hstack((np.linspace(0, -sag/2, n_secs//2+1)[1:], np.linspace(-sag/2, 0, n_secs//2+1)))
    return x, y


def get_tilted_pyramid(l, x_end, y_end, n_secs):
    """"Tilt the discretized pyramid of `get_pyramid` around origin.

    Args:
        l (float): Length of upper sides.
        x_end (float): x-position of last point.
        y_end (float): y-position of last point.
        n_secs (int): Number of equal length elements to discretize the sides in.

    Returns:
        tuple: x,y-coordinates of discretized tilted pyramid - excluding point at origin.

    """
    base = (x_end**2 + y_end**2)**.5
    xt, yt = get_pyramid(l, base, n_secs)

    theta = np.arctan2(y_end, x_end)
    r = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    x, y = [], []
    for xti, yti in zip(xt, yt):
        xi, yi = r.dot(np.array([[xti], [yti]]))
        x.append(xi)
        y.append(yi)

    return np.hstack(x), np.hstack(y)


def get_moving_pyramid(l, x_end, y_end, dl, dr, n_secs, theta_r=0, dt=.01):
    """"Use finite difference approach to get speeds when moving from one pyramid shape to another for a given time
    step.

    Args:
        l (float): Length of upper sides.
        x_end (float): x-position of last point.
        y_end (float): y-position of last point.
        dl (float): Speed of length change of upper sides.
        dr (float): Speed of last point along tilted axis set by `theta_r`. When latter is set to zero, `dr` is equal to
            the speed along x.
        n_secs (int): Number of equal length elements to discretize the sides in.
        theta_r (float): Angle of tilted axis w.r.t. the x-axis. The former axis is used for defining how the pyramid
            deforms.

    Returns:
        tuple: x,y-coordinates of points and speeds of moving pyramid.

    """
    x0, y0 = get_tilted_pyramid(l, x_end, y_end, n_secs)
    xf, yf = get_tilted_pyramid(l + dl * dt, x_end + dr * np.cos(theta_r) * dt,
                                y_end + dr * np.sin(theta_r) * dt, n_secs)
    vx = (xf-x0)/dt
    vy = (yf-y0)/dt
    return x0, y0, vx, vy