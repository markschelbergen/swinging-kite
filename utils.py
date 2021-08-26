import numpy as np
import pandas as pd


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


def tranform_to_wind_rf(row, cols_in, cols_out, upwind_direction):
    x, y = row[cols_in[0]], row[cols_in[1]]
    phi = -upwind_direction-np.pi/2.
    rm = np.array([
        [np.cos(phi), np.sin(phi)],
        [-np.sin(phi), np.cos(phi)],
    ])
    return pd.Series(rm.dot(np.array([x, y])), index=cols_out)


def plot_flight_sections(ax, df):
    y0, y1 = ax.get_ylim()
    ax.set_ylim([y0, y1])
    ax.fill_between(df.time, y0, y1, where=df['pattern_section'] == 2, facecolor='lightgrey', alpha=0.5)
    ax.fill_between(df.time, y0, y1, where=df['pattern_section'] == 0, facecolor='lightsteelblue', alpha=0.5)
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

    return pitch_spsi2b, roll_spsi2b


def read_and_transform_flight_data():
    yr, m, d = 2019, 10, 8
    i_cycle = 65
    file_name = '{:d}{:02d}{:02d}_{:04d}.csv'.format(yr, m, d, i_cycle)

    df = pd.read_csv(file_name)
    # print(list(df))
    df = df[255:625]
    df['time'] = df['time'] - df['time'].iloc[0]
    df = df.interpolate()
    df.ground_tether_force = df.ground_tether_force * 9.81

    with open('20191008_{:04d}_rpy.npy'.format(i_cycle), 'rb') as f:
        rpy = np.load(f)
        df['roll'] = rpy[df.index[0]:df.index[-1]+1, 0]
        df['pitch'] = rpy[df.index[0]:df.index[-1]+1, 1]
        df['yaw'] = rpy[df.index[0]:df.index[-1]+1, 2]
    df['pitch_tau'], df['roll_tau'] = calc_rpy_wrt_tangential_plane(df)

    df['rz'] = df['kite_height']
    df['vz'] = -df['kite_0_vz']
    df['ax'], df['ay'], df['az'] = np.gradient(df['kite_0_vy'])/.1, np.gradient(df['kite_0_vx'])/.1,\
                                   -np.gradient(df['kite_0_vz'])/.1

    upwind_direction = df.loc[df.index[0], 'est_upwind_direction']
    df[['rx', 'ry']] = df.apply(tranform_to_wind_rf, args=(['kite_pos_east', 'kite_pos_north'], ['rx', 'ry'],
                                                           upwind_direction), axis=1)
    df[['vx', 'vy']] = df.apply(tranform_to_wind_rf, args=(['kite_0_vy', 'kite_0_vx'], ['vx', 'vy'],
                                                           upwind_direction), axis=1)
    df[['ax', 'ay']] = df.apply(tranform_to_wind_rf, args=(['ax', 'ay'], ['ax', 'ay'],
                                                           upwind_direction), axis=1)

    df['radius'] = np.sum(df[['rx', 'ry', 'rz']].values**2, axis=1)**.5

    return df
