import numpy as np


def calc_cartesian_coords_enu(az, el, r):
    z = np.sin(el)*r
    r_xy = (r**2 - z**2)**.5
    x = np.cos(az)*r_xy
    y = np.sin(az)*r_xy
    return np.array((x, y, z))


def plot_vector(p0, v, ax, scale_vector=2, color='g', label=None):
    p1 = p0 + v * scale_vector
    vector = np.vstack(([p0], [p1])).T
    ax.plot3D(vector[0], vector[1], vector[2], color=color, label=label)


def unravel_euler_angles(rm, sequence='321', elevation_ref=None, azimuth_ref=None):
    # Extracting orientation angles from rotation matrix, use inertial (e.g. earth) to body reference frame rotation
    # matrix as input!

    if azimuth_ref is not None:
        rm_e2w = np.array([
            [np.cos(azimuth_ref), np.sin(azimuth_ref), 0],
            [-np.sin(azimuth_ref), np.cos(azimuth_ref), 0],
            [0, 0, 1]
        ])
        rm = rm.dot(rm_e2w.T)  # Reference to body

    if elevation_ref is not None:
        polar_angle = np.pi/2 - elevation_ref
        rm_e2ref = np.array([
            [np.cos(polar_angle), 0, -np.sin(polar_angle)],
            [0, 1, 0],
            [np.sin(polar_angle), 0, np.cos(polar_angle)]
        ])
        rm = rm.dot(rm_e2ref.T)  # Reference to body

    if sequence == '321':
        yaw = np.arctan2(rm[0, 1], rm[0, 0])
        pitch = -np.arctan2(rm[0, 2], (rm[1, 2] ** 2 + rm[2, 2] ** 2) ** .5)
        roll = np.arctan2(rm[1, 2], rm[2, 2])
    elif sequence == '231':
        pitch = -np.arctan2(rm[0, 2], rm[0, 0])
        roll = -np.arctan2(rm[2, 1], rm[1, 1])
        yaw = np.arctan2(rm[0, 1], (rm[0, 0] ** 2 + rm[0, 2] ** 2) ** .5)
    elif sequence == '312':
        yaw = -np.arctan2(rm[1, 0], rm[1, 1])
        pitch = -np.arctan2(rm[0, 2], rm[2, 2])
        roll = np.arctan2(rm[1, 2], (rm[1, 0] ** 2 + rm[1, 1] ** 2) ** .5)
    elif sequence == '213':
        yaw = np.arctan2(rm[0, 1], rm[1, 1])
        pitch = np.arctan2(rm[2, 0], rm[2, 2])
        roll = -np.arctan2(rm[2, 1], (rm[0, 1] ** 2 + rm[1, 1] ** 2) ** .5)
    elif sequence == '123':
        yaw = -np.arctan2(rm[1, 0], rm[0, 0])
        pitch = np.arctan2(rm[2, 0], (rm[1, 0] ** 2 + rm[0, 0] ** 2) ** .5)
        roll = -np.arctan2(rm[2, 1], rm[2, 2])
    elif sequence == '132':
        yaw = -np.arctan2(rm[1, 0], (rm[2, 0] ** 2 + rm[0, 0] ** 2) ** .5)
        pitch = np.arctan2(rm[2, 0], rm[0, 0])
        roll = np.arctan2(rm[1, 2], rm[1, 1])
    else:
        raise ValueError("Invalid angle sequence provided.")

    return yaw, pitch, roll