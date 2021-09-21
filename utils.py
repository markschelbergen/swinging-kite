import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from find_consistent_kite_states import find_acceleration_matching_kite_trajectory


def calc_cartesian_coords_enu(az, el, r):
    z = np.sin(el)*r
    r_xy = (r**2 - z**2)**.5
    x = np.cos(az)*r_xy
    y = np.sin(az)*r_xy
    return np.array((x, y, z))


def calc_spherical_coords(x, y, z):
    az = np.arctan2(y, x)
    r = (x**2 + y**2 + z**2)**.5
    el = np.arcsin(z/r)
    return np.array((az, el, r))


def plot_vector(p0, v, ax, scale_vector=2, color=None, label=None, linestyle=None):
    p1 = p0 + v * scale_vector
    vector = np.vstack(([p0], [p1])).T
    ax.plot3D(vector[0], vector[1], vector[2], color=color, label=label, linestyle=linestyle)


def rotation_matrix_earth_sphere(phi=0., beta=np.pi/2., yaw=0.):
    # Note that the elevation angle should be 90 degrees to yield the unity matrix.
    # For kappa=0, the sphere coordinates are given in the polar, azimuth, and radial direction.
    r1 = np.array([
        [np.cos(phi), -np.sin(phi), 0],
        [np.sin(phi), np.cos(phi), 0],
        [0, 0, 1]
    ])

    r2 = np.array([
        [np.cos(np.pi/2-beta), 0, np.sin(np.pi/2-beta)],
        [0, 1, 0],
        [-np.sin(np.pi/2-beta), 0, np.cos(np.pi/2-beta)]
    ])

    r3 = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    return r1.dot(r2).dot(r3)


def plot_vector_2d(p0, v, ax, scale_vector=.01, **kwargs):
    p1 = p0 + v * scale_vector
    vector = np.vstack(([p0], [p1])).T
    ax.plot(vector[0], vector[1], **kwargs)


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


def tranform_to_wind_rf(x, y, upwind_direction):
    phi = -upwind_direction-np.pi/2.
    rm = np.array([
        [np.cos(phi), np.sin(phi)],
        [-np.sin(phi), np.cos(phi)],
    ])
    return rm.dot(np.array([x, y]))


def plot_flight_sections(ax, df):
    y0, y1 = ax.get_ylim()
    ax.set_ylim([y0, y1])
    ax.fill_between(df.time, y0, y1, where=df['flag_turn'], facecolor='lightsteelblue', alpha=0.5) # lightgrey
    # for i_ph in range(6):
    #     mask = df['phase'] == i_ph
    #     if np.sum(mask) > 0:
    #         ax.axvline(df.loc[df.index[mask][0], 'time'], linestyle='--', color='grey')


def rotation_matrix_earth2body(roll, pitch, yaw, sequence='321'):
    # Returns rotation matrix to transform from earth to body reference frame.
    # Earth: East, North, up
    # Body: front, left, up

    # Rotational matrix for roll.
    r_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), np.sin(roll)],
        [0, -np.sin(roll), np.cos(roll)]
    ])

    # Rotational matrix for pitch (nose down).
    r_pitch = np.array([
        [np.cos(pitch), 0, -np.sin(pitch)],
        [0, 1, 0],
        [np.sin(pitch), 0, np.cos(pitch)]
    ])

    # Rotational matrix for yaw.
    r_yaw = np.array([
        [np.cos(yaw), np.sin(yaw), 0],
        [-np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    rs = [r_roll, r_pitch, r_yaw]
    sequence = [int(i)-1 for i in sequence]
    r = rs[sequence[2]].dot(rs[sequence[1]].dot(rs[sequence[0]]))
    return r


def calc_rpy_wrt_tangential_plane(df):
    pitch_spsi2b, yaw_spsi2b, roll_spsi2b = [], [], []

    for idx, row in df.iterrows():
        r = np.array([row['kite_pos_east'], row['kite_pos_north'], row['kite_height']])
        l12 = np.linalg.norm(r[:2])
        l = np.linalg.norm(r)
        rm_s2e = np.array([
            [r[0]*r[2]/(l*l12), -r[1]/l12,  r[0]/l],
            [r[1]*r[2]/(l*l12), r[0]/l12, r[1]/l],
            [-l12/l,              0,         r[2]/l],
        ])

        r_e2b = rotation_matrix_earth2body(row['roll'], row['pitch'], row['yaw'])
        rm_spsi2b = r_e2b.dot(rm_s2e)

        y, p, r = unravel_euler_angles(rm_spsi2b, '321')  # 312 would give virtually the same result.
        pitch_spsi2b.append(p), yaw_spsi2b.append(y), roll_spsi2b.append(r)

    return pitch_spsi2b, roll_spsi2b, yaw_spsi2b


def read_and_transform_flight_data(make_trajectory_consistent_with_integrator=False):
    from turning_center import find_turns_for_rolling_window

    # yr, m, d = 2019, 10, 8
    # i_cycle = 65
    # file_name = '{:d}{:02d}{:02d}_{:04d}.csv'.format(yr, m, d, i_cycle)
    # df = pd.read_csv(file_name)
    # with open('20191008_{:04d}_rpy.npy'.format(i_cycle), 'rb') as f:
    #     rpy = np.load(f)
    #     df['roll'] = rpy[:, 0]*180./np.pi
    #     df['pitch'] = -rpy[:, 1]*180./np.pi
    #     df['yaw'] = -rpy[:, 2]*180./np.pi + 90
    # df = df[299:513]
    #
    # # 'ground_tether_length'
    # cols = ['time', 'date', 'time_of_day', 'kite_0_vx', 'kite_0_vy', 'kite_0_vz', 'kite_1_ax', 'kite_1_ay', 'kite_1_az',
    #         'kite_0_roll', 'kite_0_pitch', 'kite_0_yaw', 'ground_tether_reelout_speed', 'ground_tether_force',
    #         'est_upwind_direction', 'kite_pos_east', 'kite_pos_north', 'kite_height',
    #         'kite_elevation', 'kite_azimuth', 'kite_distance', 'kite_actual_steering', 'roll', 'pitch', 'yaw']
    # df.to_csv("20191008_0065_fig8.csv", index=False, na_rep='nan', columns=cols)

    file_name = '20191008_0065_fig8.csv'
    df = pd.read_csv(file_name)

    df = df.interpolate()
    df['time'] = df['time'] - df['time'].iloc[0]

    df['roll'] = df.roll*np.pi/180.
    df['pitch'] = -df.pitch*np.pi/180.
    df['yaw'] = -(df.yaw-90.)*np.pi/180.
    # df['roll'] = (df.kite_0_roll-8.5)*np.pi/180.
    # df['pitch'] = (-df.kite_0_pitch+3)*np.pi/180.
    # df['yaw'] = -(df.kite_0_yaw-90.)*np.pi/180.

    df.kite_azimuth = -df.kite_azimuth
    df.ground_tether_force = df.ground_tether_force * 9.81

    df['pitch_tau'], df['roll_tau'], df['yaw_tau'] = calc_rpy_wrt_tangential_plane(df)

    df['rz'] = df['kite_height']
    df['vz'] = -df['kite_0_vz']
    df['kite_1_az'] = -df['kite_1_az']

    upwind_direction = df.loc[df.index[0], 'est_upwind_direction']
    df['rx'], df['ry'] = tranform_to_wind_rf(df['kite_pos_east'], df['kite_pos_north'], upwind_direction)
    df['vx'], df['vy'] = tranform_to_wind_rf(df['kite_0_vy'], df['kite_0_vx'], upwind_direction)
    df['kite_1_ax'], df['kite_1_ay'] = tranform_to_wind_rf(df['kite_1_ay'], df['kite_1_ax'], upwind_direction)

    # Infer kite acceleration from measurements.
    x_kite, a_kite = find_acceleration_matching_kite_trajectory(df)
    # np.save('kite_states.npy', np.hstack((x_kite, np.vstack((a_kite, [[np.nan]*3])))))
    # x_kite, a_kite = np.load('kite_states.npy')[:, :6], np.load('kite_states.npy')[:-1, 6:]

    if make_trajectory_consistent_with_integrator:
        df[['rx', 'ry', 'rz']] = x_kite[:, :3]
        df['kite_azimuth'], df['kite_elevation'], df['kite_distance'] = calc_spherical_coords(df['rx'], df['ry'], df['rz'])
        df[['vx', 'vy', 'vz']] = x_kite[:, 3:6]

    df['ax'] = np.hstack((a_kite[:, 0], [np.nan]))
    df['ay'] = np.hstack((a_kite[:, 1], [np.nan]))
    df['az'] = np.hstack((a_kite[:, 2], [np.nan]))

    return df
