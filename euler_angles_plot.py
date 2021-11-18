import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from utils import rotation_matrix_earth2body, add_panel_labels


def plot_circle(roll, pitch, yaw, axis, radius, theta_max=2*np.pi, **kwargs):
    rm_b2e = rotation_matrix_earth2body(roll, pitch, yaw).T

    n_elem = abs(int(theta_max*180/np.pi))
    theta = np.linspace(0, theta_max, n_elem)
    s = radius*np.sin(theta)
    c = radius*np.cos(theta)
    if axis == 0:  # TODO: not checked z rotation
        xyz = np.vstack([[-s], [c]])
    else:
        xyz = np.vstack([[s], [c]])
    xyz = np.insert(xyz, axis, 0, axis=0)

    xyz = rm_b2e.dot(xyz)

    plt.plot(xyz[0, :], xyz[1, :], xyz[2, :], **kwargs)


def plot_ground_square(roll, pitch, yaw, **kwargs):
    xyz = np.zeros((3, 5))
    xyz[:2, :] = get_square()

    rm = rotation_matrix_earth2body(roll, pitch, yaw).T
    xyz = rm.dot(xyz)
    plt.plot(xyz[0, :], xyz[1, :], xyz[2, :], **kwargs)


def get_square(side_x=.5, side_y=1.):
    xy = np.zeros((2, 5))
    xy[:, 0] = [side_x, side_y]
    xy[:, 1] = [-side_x, side_y]
    xy[:, 2] = [-side_x, -side_y]
    xy[:, 3] = [side_x, -side_y]
    xy[:, 4] = [side_x, side_y]

    return xy


def plot_tangential_plane(center_point, elevation, azimuth, kappa, axis=2, **kwargs):
    rm = rotation_matrix_earth2body(0, np.pi/2-elevation, azimuth).T

    # Plot rotation circle
    num = abs(int(kappa*180/np.pi))
    theta = np.linspace(0, kappa, num)
    s = 1.5*np.sin(theta)
    c = 1.5*np.cos(theta)
    xyz = np.vstack([[c], [s]])
    xyz = np.insert(xyz, 2, 0, axis=0)
    xyz = rm.dot(xyz) + np.repeat([center_point], num, axis=0).T
    plt.plot(xyz[0, :], xyz[1, :], xyz[2, :])

    # Plot square
    rm = rotation_matrix_earth2body(0, 0, kappa).dot(rm.T).T
    xyz = get_square()
    xyz = np.insert(xyz, axis, 0, axis=0)
    xyz = rm.dot(xyz) + np.repeat([center_point], 5, axis=0).T
    plt.plot(xyz[0, :], xyz[1, :], xyz[2, :], **kwargs)

    # Plot x-axis
    xaxis = np.zeros((3, 2))
    xaxis[0, 1] = 2
    xaxis = rm.dot(xaxis) + np.repeat([center_point], 2, axis=0).T
    plt.plot(xaxis[0, :], xaxis[1, :], xaxis[2, :], **kwargs)


def plot_ref_frame(roll, pitch, yaw, plot_axes=range(3), length=1, **kwargs):
    rm_b2e = rotation_matrix_earth2body(roll, pitch, yaw).T
    for i in plot_axes:
        v = np.zeros(3)
        v[i] = length
        v = rm_b2e.dot(v)
        plt.plot([0, v[0]], [0, v[1]], [0, v[2]], **kwargs)


def plot_rot_axis(roll, pitch, yaw, axis, **kwargs):
    rm_b2e = rotation_matrix_earth2body(roll, pitch, yaw).T
    v = np.zeros(3)
    v[axis] = 1.3
    v = rm_b2e.dot(v)
    plt.plot([-v[0], v[0]], [-v[1], v[1]], [-v[2], v[2]], linestyle='--', linewidth=.5, **kwargs)


def plot_euler_sequence(only_pitch=False, ax3d=None):
    if ax3d is None:
        ax3d = plt.subplot(111, projection='3d', proj_type='ortho')
    ax3d.axis('off')
    ax3d.set_xlim([-1/1.5, 1/1.5])
    ax3d.set_ylim([-1/1.5, 1/1.5])
    ax3d.set_zlim([-1/1.5, 1/1.5])
    # # https://delftxtools.tudelft.nl/AE4T40_Airborne_Wind_Energy/threejs/kiteV3_static.html
    # # roll=0, pitch=-16, yaw=-149
    # ax3d.azim = 109
    # ax3d.elev = 62
    ax3d.azim = -126
    ax3d.elev = 49

    pitch = 15*np.pi/180.
    roll = -30*np.pi/180.
    r, p = 0, 0
    colors = ['k', 'b', 'g']
    plot_axes = [2]

    for i, (rot_axis, clr, theta) in enumerate(zip([2, 1, 0], colors, [None, pitch, roll])):
        if theta is not None:
            plot_circle(r, p, 0, rot_axis, 1, theta_max=theta, color=colors[i-1], linewidth=2)
        if i == 1:
            p = pitch
        elif i == 2:
            r = roll
        if i > 0:
            plot_circle(r, p, 0, rot_axis, 1, color=colors[i-1], linewidth=.3)
            plot_rot_axis(r, p, 0, rot_axis, color=colors[i-1])
        if i == 0:
            rot_axis = None
        # if i == 2:
        #     plot_axes = range(3)
        plot_ref_frame(r, p, 0, plot_axes=plot_axes, color=clr)

        if only_pitch and i == 1:
            break

    plot_ground_square(0, 0, 0, color='k', linewidth=1.)
    plot_ground_square(r, p, 0, color=clr, linewidth=.5)


def plot_small_earth(**kwargs):
    elevation0 = 30*np.pi/180.
    radius = 10
    plt.figure(figsize=[5.4, 3.2])
    plt.subplots_adjust(left=0., right=1., bottom=-.5, top=1.15)
    ax3d = plt.subplot(111, projection='3d')#, proj_type='ortho')
    ax3d.azim = 42
    ax3d.elev = 22
    plt.axis('off')
    ax3d.set_xlim([-radius/1.5, radius/1.5])
    ax3d.set_ylim([-radius/1.5, radius/1.5])
    ax3d.set_zlim([-radius/1.5, radius/1.5])

    axis_length = 12
    az_we = 140*np.pi/180
    plot_ref_frame(0, 0, az_we, length=axis_length, color='grey')
    plot_ref_frame(0, 0, 0, length=axis_length, color='k')

    # Lissajous
    t = np.linspace(0, 2*np.pi, 361)
    phi_liss = 35*np.pi/180.*np.sin(t)
    beta_liss = 10*np.pi/180.*np.sin(2*t)+elevation0

    elevation = beta_liss[25]
    azimuth = phi_liss[25]

    x = radius*np.cos(phi_liss)*np.cos(beta_liss)
    y = radius*np.sin(phi_liss)*np.cos(beta_liss)
    z = radius*np.sin(beta_liss)
    xyz = np.vstack([[x], [y], [z]])

    plt.plot(xyz[0, :], xyz[1, :], xyz[2, :], linewidth=.5, color='grey')

    # Ground circle
    theta = np.linspace(0, 2*np.pi, 361)
    s = radius*np.sin(theta)
    c = radius*np.cos(theta)
    xyz = np.vstack([[c], [s]])
    xyz = np.insert(xyz, 2, 0, axis=0)
    plt.plot(xyz[0, :], xyz[1, :], xyz[2, :], **kwargs)

    # Elevated circle
    r = np.cos(elevation)*radius
    s = r*np.sin(theta)
    c = r*np.cos(theta)
    xyz = np.vstack([[s], [c]])
    z = np.sin(elevation)*radius
    xyz = np.insert(xyz, 2, z, axis=0)
    plt.plot(xyz[0, :], xyz[1, :], xyz[2, :], **kwargs)

    # Azimuth earth to wind
    phi = np.linspace(0, az_we, abs(int(az_we*180/np.pi)))
    s = .8*radius*np.sin(phi)
    c = .8*radius*np.cos(phi)
    xyz = np.vstack([[c], [s]])
    xyz = np.insert(xyz, 2, 0, axis=0)
    plt.plot(xyz[0, :], xyz[1, :], xyz[2, :], color='grey')

    ref_line = np.zeros((3, 2))
    ref_line[0, 1] = radius*np.cos(azimuth)
    ref_line[1, 1] = radius*np.sin(azimuth)
    plt.plot(ref_line[0, :], ref_line[1, :], ref_line[2, :], **kwargs)

    # Azimuth
    phi = np.linspace(0, azimuth, abs(int(azimuth*180/np.pi)))
    s = radius*np.sin(phi)
    c = radius*np.cos(phi)
    xyz = np.vstack([[c], [s]])
    xyz = np.insert(xyz, 2, 0, axis=0)
    plt.plot(xyz[0, :], xyz[1, :], xyz[2, :], linewidth=1.5)

    ref_line = np.zeros((3, 2))
    ref_line[0, 1] = radius*np.cos(azimuth)
    ref_line[1, 1] = radius*np.sin(azimuth)
    plt.plot(ref_line[0, :], ref_line[1, :], ref_line[2, :], **kwargs)

    # Elevation
    beta = np.linspace(0, elevation, abs(int(elevation*180/np.pi)))
    x = radius*np.cos(azimuth)*np.cos(beta)
    y = radius*np.sin(azimuth)*np.cos(beta)
    z = radius*np.sin(beta)
    xyz = np.vstack([[x], [y], [z]])
    plt.plot(xyz[0, :], xyz[1, :], xyz[2, :], linewidth=1.5)

    ref_line = np.zeros((3, 2))
    ref_line[0, 1] = radius*np.cos(azimuth)*np.cos(elevation)
    ref_line[1, 1] = radius*np.sin(azimuth)*np.cos(elevation)
    ref_line[2, 1] = radius*np.sin(elevation)
    plt.plot(ref_line[0, :], ref_line[1, :], ref_line[2, :], **kwargs)

    # Half circles
    theta = np.linspace(0, np.pi, 181)
    s = radius*np.sin(theta)
    c = radius*np.cos(theta)
    for plane in [0, 1]:
        xyz = np.vstack([[c], [s]])
        xyz = np.insert(xyz, plane, 0, axis=0)
        plt.plot(xyz[0, :], xyz[1, :], xyz[2, :], **kwargs)

    center_point_square = [radius*np.cos(azimuth)*np.cos(elevation),
                           radius*np.sin(azimuth)*np.cos(elevation),
                           radius*np.sin(elevation)]

    plot_tangential_plane(center_point_square, elevation, azimuth, 1.5*np.pi/2, color='k')

plt.figure(figsize=[6.4, 2.6])
plt.subplots_adjust(top=1.2, bottom=-.20, left=0.08, right=1.0, hspace=0.0, wspace=0.2)
ax0 = plt.subplot(121, projection='3d', proj_type='ortho')
plot_euler_sequence(only_pitch=True, ax3d=ax0)
ax1 = plt.subplot(122, projection='3d', proj_type='ortho')
plot_euler_sequence(ax3d=ax1)
add_panel_labels(np.array([ax0, ax1]))

plot_small_earth(color='k', linestyle='--', linewidth=.7)
plt.show()