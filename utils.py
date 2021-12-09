import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flight_trajectory_reconstruction import find_acceleration_matching_kite_trajectory
from os.path import isfile

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


def rotation_matrix_earth2sphere(phi=0., beta=np.pi/2., heading=0.):
    # Note that the elevation angle should be 90 degrees to yield the unity matrix.
    # For kappa=0, the sphere coordinates are given in the polar, azimuth, and radial direction.
    r1 = np.array([
        [np.cos(phi), np.sin(phi), 0],
        [-np.sin(phi), np.cos(phi), 0],
        [0, 0, 1]
    ])

    r2 = np.array([
        [np.sin(beta), 0, -np.cos(beta)],
        [0, 1, 0],
        [np.cos(beta), 0, np.sin(beta)]
    ])

    r3 = np.array([
        [np.cos(heading), np.sin(heading), 0],
        [-np.sin(heading), np.cos(heading), 0],
        [0, 0, 1]
    ])

    return r3.dot(r2).dot(r1)


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


def tranform_to_wind_rf(x, y, phi):
    rm_we = np.array([
        [np.cos(phi), np.sin(phi)],
        [-np.sin(phi), np.cos(phi)],
    ])
    return rm_we.dot(np.array([x, y]))


def plot_flight_sections(ax, df):
    y0, y1 = ax.get_ylim()
    ax.set_ylim([y0, y1])
    ax.fill_between(df.time, y0, y1, where=df['flag_turn'], facecolor='lightsteelblue', alpha=0.5) # lightgrey


def plot_flight_sections2(ax, df, use_flag_turn=False):
    if isinstance(ax, np.ndarray):
        ax = ax.reshape(-1)
    else:
        ax = [ax]
    for a in ax:
        y0, y1 = a.get_ylim()
        a.set_ylim([y0, y1])
        if use_flag_turn:
            a.fill_between(df.time, y0, y1, where=df['flag_turn'], facecolor='lightsteelblue', alpha=0.5) # lightgrey
        else:
            a.fill_between(df.time, y0, y1, where=df['pattern_section'] == 2, facecolor='lightgrey', alpha=0.5)
            a.fill_between(df.time, y0, y1, where=df['pattern_section'] == 0, facecolor='lightsteelblue', alpha=0.5)
            for i_ph in range(6):
                mask = df['phase'] == i_ph
                if mask.sum() > 0:
                    a.axvline(df.loc[mask.idxmax(), 'time'], linestyle='--', color='grey')


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


def calc_rpy_bridle_wrt_tangential_plane(df, rpy_cols=('roll', 'pitch', 'yaw')):
    roll_s2b, pitch_s2b, yaw_s2b = [], [], []

    for idx, row in df.iterrows():
        r = np.array([row['kite_pos_east'], row['kite_pos_north'], row['kite_height']])
        l12 = np.linalg.norm(r[:2])
        l = np.linalg.norm(r)
        rm_es = np.array([
            [r[0]*r[2]/(l*l12), -r[1]/l12,  r[0]/l],
            [r[1]*r[2]/(l*l12), r[0]/l12, r[1]/l],
            [-l12/l,              0,         r[2]/l],
        ])

        rm_ce = rotation_matrix_earth2body(row[rpy_cols[0]], row[rpy_cols[1]], row[rpy_cols[2]])
        rm_bc = rotation_matrix_earth2body(0, -get_pitch_nose_down_angle_v3(1-row['kite_actual_depower']/100.), 0)
        rm_bs = rm_bc.dot(rm_ce).dot(rm_es)

        y, p, r = unravel_euler_angles(rm_bs, '321')  # 312 would give virtually the same result.
        roll_s2b.append(r), pitch_s2b.append(p), yaw_s2b.append(y)

    return roll_s2b, pitch_s2b, yaw_s2b


def read_and_transform_flight_data(make_kinematics_consistent=True, i_cycle=None):  #, kite_states_file_suffix=''):
    from turning_center import find_turns_for_rolling_window

    if i_cycle is not None:
        yr, m, d = 2019, 10, 8
        folder = '/home/mark/Projects/quasi-steady-model-sandbox/flight_data/cycles/{:d}{:02d}{:02d}v2/'.format(yr, m, d)
        file_name = folder + '{:d}{:02d}{:02d}_{:04d}.csv'.format(yr, m, d, i_cycle)
        df = pd.read_csv(file_name)

        try:
            with open(folder + '20191008_{:04d}_rpy_v2.npy'.format(i_cycle), 'rb') as f:
                rpy = np.load(f)
                df['roll'] = rpy[:, 0]*180./np.pi
                df['pitch'] = -rpy[:, 1]*180./np.pi
                df['yaw'] = -rpy[:, 2]*180./np.pi + 90
        except FileNotFoundError:
            df['roll'], df['pitch'], df['yaw'] = np.nan, np.nan, np.nan
        df['time'] = df['time'] - df['time'].iloc[0]
        # df = df[299:513]
        #
        # cols = ['time', 'date', 'time_of_day', 'kite_0_vx', 'kite_0_vy', 'kite_0_vz', 'kite_1_ax', 'kite_1_ay', 'kite_1_az',
        #         'kite_0_roll', 'kite_0_pitch', 'kite_0_yaw', 'kite_1_roll', 'kite_1_pitch', 'kite_1_yaw', 'ground_tether_reelout_speed', 'ground_tether_force',
        #         'est_upwind_direction', 'kite_pos_east', 'kite_pos_north', 'kite_height',
        #         'kite_elevation', 'kite_azimuth', 'kite_distance', 'kite_heading', 'kite_course', 'kite_actual_steering', 'roll', 'pitch', 'yaw', 'kite_actual_depower']
        # df.to_csv("20191008_0065_fig8.csv", index=False, na_rep='nan', columns=cols)
    else:
        file_name = '20191008_0065_fig8.csv'
        df = pd.read_csv(file_name)
    df['time'] = df['time'].round(1)
    df = df.interpolate()

    # Lower only used for aero force decomposition
    df['roll'] = df.roll*np.pi/180.
    df['pitch'] = -df.pitch*np.pi/180.
    df['yaw'] = -(df.yaw-90.)*np.pi/180.

    df['roll0'] = (df.kite_0_roll-8.5)*np.pi/180.
    df['pitch0'] = (-df.kite_0_pitch+7)*np.pi/180.
    df['yaw0'] = -(df.kite_0_yaw-90.)*np.pi/180.

    df['roll1'] = (df.kite_1_roll-8.5)*np.pi/180.
    df['pitch1'] = (-df.kite_1_pitch+7)*np.pi/180.
    df['yaw1'] = -(df.kite_1_yaw-90.)*np.pi/180.

    # df.kite_1_yaw_rate = -df.kite_1_yaw_rate

    df.kite_azimuth = -df.kite_azimuth
    df.ground_tether_force = df.ground_tether_force * 9.81

    find_turns_for_rolling_window(df)

    df['roll_tau'], df['pitch_tau'], df['yaw_tau'] = calc_rpy_bridle_wrt_tangential_plane(df, rpy_cols=['roll', 'pitch', 'yaw'])
    df['roll0_tau'], df['pitch0_tau'], df['yaw0_tau'] = calc_rpy_bridle_wrt_tangential_plane(df, rpy_cols=['roll0', 'pitch0', 'yaw0'])
    df['roll1_tau'], df['pitch1_tau'], df['yaw1_tau'] = calc_rpy_bridle_wrt_tangential_plane(df, rpy_cols=['roll1', 'pitch1', 'yaw1'])

    df['rz'] = df['kite_height']
    df['vz'] = -df['kite_0_vz']
    df['kite_1_az'] = -df['kite_1_az']

    phi_upwind_direction = -df.loc[df.index[0], 'est_upwind_direction']-np.pi/2.
    df['rx'], df['ry'] = tranform_to_wind_rf(df['kite_pos_east'], df['kite_pos_north'], phi_upwind_direction)
    df['vx'], df['vy'] = tranform_to_wind_rf(df['kite_0_vy'], df['kite_0_vx'], phi_upwind_direction)
    df[['kite_0_vx', 'kite_0_vy', 'kite_0_vz']] = df[['vx', 'vy', 'vz']].copy()
    df['kite_1_ax'], df['kite_1_ay'] = tranform_to_wind_rf(df['kite_1_ay'], df['kite_1_ax'], phi_upwind_direction)

    if i_cycle is None:
        kite_states_file = 'kite_states_cycle65.npy'
        x, u = np.load(kite_states_file)[299:513, :8], np.load(kite_states_file)[299:513-1, 8:]
    else:
        kite_states_file = 'kite_states_cycle{}.npy'.format(i_cycle)
        if isfile(kite_states_file):
            x, u = np.load(kite_states_file)[:, :8], np.load(kite_states_file)[:-1, 8:]
        else:
            x, u = find_acceleration_matching_kite_trajectory(df)
            np.save(kite_states_file, np.hstack((x, np.vstack((u, [[np.nan]*4])))))
    df['ddl'] = np.hstack((u[:, 3], [np.nan]))
    df['dl'] = x[:, 7]
    df['l'] = x[:, 3]
    x_kite = np.delete(x, [3, 7], axis=1)
    a_kite = np.delete(u, 3, axis=1)

    if make_kinematics_consistent:
        # Not just use the inferred acceleration, but also impose the corresponding position and velocity.
        df[['rx', 'ry', 'rz']] = x_kite[:, :3]
        df['kite_azimuth'], df['kite_elevation'], df['kite_distance'] = calc_spherical_coords(df['rx'], df['ry'], df['rz'])
        df[['vx', 'vy', 'vz']] = x_kite[:, 3:6]

    df['ax'] = np.hstack((a_kite[:, 0], [np.nan]))
    df['ay'] = np.hstack((a_kite[:, 1], [np.nan]))
    df['az'] = np.hstack((a_kite[:, 2], [np.nan]))

    # from mpl_toolkits import mplot3d
    # plt.figure()
    # # plt.plot(df['phase'])
    # # plt.plot(df['kite_actual_depower'])
    # ax3d = plt.axes(projection='3d')
    # ax3d.plot3D(x_kite[:, 0], x_kite[:, 1], x_kite[:, 2])
    # plt.show()

    return df


def add_panel_labels(ax, offset_x=.15):
    import string
    if ax.ndim > 1:
        denominator = ax.shape[1]
        ax = ax.reshape(-1)
    else:
        denominator = ax.shape[0]
    for i, a in enumerate(ax):
        label = '('+string.ascii_lowercase[i]+')'
        if isinstance(offset_x, float):
            x = -offset_x
        else:
            x = -offset_x[i % denominator]
        if hasattr(a, 'text2D'):
            fun = a.text2D
        else:
            fun = a.text
        fun(x, .5, label, transform=a.transAxes, fontsize='large')  #, fontweight='bold', va='top', ha='right')


def get_pitch_nose_down_angle_v3(u_p):
    # Determine the pitch down angle (see fig. 5 in Oehler) as a function of the power setting.
    # u_p is the power setting ranging from 0 (fully depowered) - 1 (fully powered). The chord is assumed to be
    # perpendicular to the power line when u_p=u_p_ref.
    # Flight test of 8-10-2019, u_p=.78 for reel-out and u_p=.7 for reel-in.
    # Powered flight (u_p=.78) pitch down angle: 3.2 degrees
    # Depowered flight (u_p=.7) pitch down angle: 9.8 degrees

    # Expression adopted from Oehler eq. 5.
    b = 11.414  # Length between bridle point and front bridle connection with kite.
    c = 1.81  # Chordwise length between front and rear bridle connection.
    a0 = (b**2+c**2)**.5  # Length between bridle point and rear bridle connection with kite (trailing edge?).

    dl_du = 5.  # Depower tape length range between fully depowered and powered. (1.7 used by Oehler)
    u_p_ref = .82  # Power setting at which the power lines are perpendicular to the chord.
    da = (u_p_ref - u_p)*dl_du/2
    a = a0+da

    pitch = np.arccos((b**2+c**2-a**2)/(2*b*c))-np.pi/2

    return pitch


if __name__ == "__main__":
    print(get_pitch_nose_down_angle_v3(.78)*180./np.pi)