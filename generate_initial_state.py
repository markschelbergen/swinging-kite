import numpy as np
import casadi as ca


def get_pyramid(l, b, n_elements):
    """"Create pyramid-like shape with fixed length of (upper) sides discretized in equal length elements.

    Args:
        l (float): Total length of sides.
        b (float): Length of base.
        n_elements (int): Number of equal length elements to discretize the sides in.

    Returns:
        tuple: x,y-coordinates of discretized pyramid - excluding point at origin.

    """
    if n_elements % 2 == 0:
        sag = (l ** 2 - b ** 2) ** .5
        x = np.linspace(0, b, n_elements + 1)[1:]
        y = np.hstack((np.linspace(0, -sag / 2, n_elements // 2 + 1)[1:], np.linspace(-sag / 2, 0, n_elements // 2 + 1)[1:]))
    else:
        li = l / n_elements
        sag = ((l-li) ** 2 - (b - li) ** 2) ** .5
        x = np.hstack((np.linspace(0, (b - li) / 2, n_elements // 2 + 1)[1:],
                       np.linspace((b - li) / 2 + li, b, n_elements // 2 + 1)))
        y = np.hstack((np.linspace(0, -sag / 2, n_elements // 2 + 1)[1:], np.linspace(-sag / 2, 0, n_elements // 2 + 1)))
    return x, y


def get_tilted_pyramid(l, x_end, y_end, n_elements):
    """"Tilt the discretized pyramid of `get_pyramid` around origin.

    Args:
        l (float): Length of upper sides.
        x_end (float): x-position of last point.
        y_end (float): y-position of last point.
        n_elements (int): Number of equal length elements to discretize the sides in.

    Returns:
        tuple: x,y-coordinates of discretized tilted pyramid - excluding point at origin.

    """
    base = (x_end**2 + y_end**2)**.5
    xt, yt = get_pyramid(l, base, n_elements)

    theta = np.arctan2(y_end, x_end)
    r = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    x, y = [], []
    for xti, yti in zip(xt, yt):
        xi, yi = r.dot(np.array([[xti], [yti]]))
        x.append(xi)
        y.append(yi)

    return np.hstack(x), np.hstack(y)


def get_tilted_pyramid_3d(l, x_end, y_end, z_end, n_elements):
    azimuth = np.arctan2(y_end, x_end)
    r_xy_end = (x_end**2 + y_end**2)**.5
    r_xy, z = get_tilted_pyramid(l, r_xy_end, z_end, n_elements)
    x = np.cos(azimuth)*r_xy
    y = np.sin(azimuth)*r_xy
    return x, y, z


def get_moving_pyramid(l, x_end, y_end, dl, dr, n_elements, theta_r=0, dt=.01):
    """"Use finite difference approach to get speeds when moving from one pyramid shape to another with speed along a
    given axis and for a given time step.

    Args:
        l (float): Length of upper sides.
        x_end (float): x-position of last point.
        y_end (float): y-position of last point.
        dl (float): Speed of length change of upper sides.
        dr (float): Speed of last point along tilted axis set by `theta_r`. When latter is set to zero, `dr` is equal to
            the speed along x.
        n_elements (int): Number of equal length elements to discretize the sides in.
        theta_r (float): Angle of tilted axis w.r.t. the x-axis. The former axis is used for defining how the pyramid
            deforms.

    Returns:
        tuple: x,y-coordinates of points and speeds of moving pyramid.

    """
    x0, y0 = get_tilted_pyramid(l, x_end, y_end, n_elements)
    xf, yf = get_tilted_pyramid(l + dl * dt, x_end + dr * np.cos(theta_r) * dt,
                                y_end + dr * np.sin(theta_r) * dt, n_elements)
    vx = (xf-x0)/dt
    vy = (yf-y0)/dt
    return x0, y0, vx, vy


def check_constraints(dyn, x0):
    cons_x0 = ca.Function('f', [dyn['x']], [dyn['g'], dyn['dg']])(x0)
    max_g = np.amax(np.abs(np.array(cons_x0[0])))
    print("Max. tether length constraints: {:.2E}".format(max_g))
    max_dg = np.amax(np.abs(np.array(cons_x0[1])))
    print("Max. tether speed constraints: {:.2E}".format(max_dg))
    assert max_g < 1e-5
    assert max_dg < 1e-5


def find_initial_velocities_satisfying_constraints(dyn, x0, v_end, plot=False):
    n_point_masses = (len(x0)-2)//6
    n_pm_opt = n_point_masses-1
    r_guess = x0[:n_pm_opt*3]
    r_end = x0[n_pm_opt*3:n_point_masses*3]
    v_guess = x0[n_point_masses*3:2*n_point_masses*3-3]
    l = x0[2*n_point_masses*3]
    dl = x0[2*n_point_masses*3+1]

    opti = ca.casadi.Opti()
    r = opti.variable(n_pm_opt*3)
    opti.set_initial(r, r_guess)
    v = opti.variable(n_pm_opt*3)
    opti.set_initial(v, v_guess)

    x = ca.vertcat(r, r_end, v, v_end, l, dl)
    g, dg = ca.Function('f', [dyn['x']], [dyn['g'], dyn['dg']])(x)
    opti.minimize(ca.sumsqr(g)+ca.sumsqr(dg))

    opti.solver('ipopt')
    sol = opti.solve()
    x_sol = sol.value(x)

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 12))
        ax3d = plt.axes(projection='3d')
        ax3d.set_xlim([0, 250])
        ax3d.set_ylim([-125, 125])
        ax3d.set_zlim([0, 250])
        ax3d.plot3D(x_sol[:3*n_point_masses:3], x_sol[1:3*n_point_masses:3], x_sol[2:3*n_point_masses:3], 's-')
        plt.show()
        exit()

    return x_sol
