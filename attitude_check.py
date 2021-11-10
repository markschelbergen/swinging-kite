import numpy as np
from utils import rotation_matrix_earth2sphere, plot_vector_2d, rotation_matrix_earth2body
from system_properties import vwx

rpy_suffix = '0'


def plot_apparent_wind_velocity(ax, row, vwx, scale_vector=1, **kwargs):
    v_kite = np.array(list(row[['vx', 'vy', 'vz']]))
    v_app = v_kite - np.array([vwx, 0, 0])
    r_sw = rotation_matrix_earth2sphere(row['kite_azimuth'], row['kite_elevation'], 0)

    vec_sphere = r_sw.dot(v_app)
    vec_proj = np.array([vec_sphere[1], -vec_sphere[0]])
    plot_vector_2d([row['kite_azimuth']*180./np.pi, row['kite_elevation']*180./np.pi], vec_proj, ax, scale_vector,
                   **kwargs)


def plot_ex(ax, row, phi_upwind_direction, **kwargs):
    r_sw = rotation_matrix_earth2sphere(row['kite_azimuth'], row['kite_elevation'], 0)
    r_bw = rotation_matrix_earth2body(row['roll'+rpy_suffix], row['pitch'+rpy_suffix], row['yaw'+rpy_suffix] - phi_upwind_direction)
    exb_earth = r_bw.T[:, 0]

    vec_sphere = r_sw.dot(exb_earth)
    vec_proj = np.array([vec_sphere[1], -vec_sphere[0]])
    plot_vector_2d([row['kite_azimuth']*180./np.pi, row['kite_elevation']*180./np.pi], vec_proj, ax, 1e1, **kwargs)


def plot_kite_velocity(ax, row, **kwargs):
    v_kite = np.array(list(row[['vx', 'vy', 'vz']]))
    r_sw = rotation_matrix_earth2sphere(row['kite_azimuth'], row['kite_elevation'], 0)

    vec_sphere = r_sw.dot(v_kite)
    vec_proj = np.array([vec_sphere[1], -vec_sphere[0]])
    plot_vector_2d([row['kite_azimuth']*180./np.pi, row['kite_elevation']*180./np.pi], vec_proj, ax, 1, **kwargs)


def calc_kite_front_wrt_projected_velocity(r_kite, v_kite, rm_wind2body):
    ez_v = r_kite/np.linalg.norm(r_kite)
    ey_v = np.cross(ez_v, v_kite)/np.linalg.norm(np.cross(ez_v, v_kite))
    ex_v = np.cross(ey_v, ez_v)
    rm_v2w_i = np.vstack(([ex_v], [ey_v], [ez_v])).T

    exb_wind = rm_wind2body.T[:, 0]
    exb_v = rm_v2w_i.T.dot(exb_wind)
    return np.arctan2(exb_v[1], exb_v[0])


def plot_velocity_and_respective_kite_attitude(flight_data):
    from turning_center import mark_points
    phi_upwind_direction = -flight_data.loc[flight_data.index[0], 'est_upwind_direction']-np.pi/2.

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(flight_data.time, np.sum(flight_data[['vx', 'vy', 'vz']]**2, axis=1)**.5)
    ax[0].set_ylabel('Kite speed [m/s]')

    kite_front_wrt_projected_kite_velocity = np.empty(flight_data.shape[0])
    kite_front_wrt_projected_app_velocity = np.empty(flight_data.shape[0])
    for i, (idx, row) in enumerate(flight_data.iterrows()):
        r_kite = np.array(list(row[['rx', 'ry', 'rz']]))
        v_kite = np.array(list(row[['vx', 'vy', 'vz']]))
        r_bw = rotation_matrix_earth2body(row['roll'+rpy_suffix], row['pitch'+rpy_suffix], row['yaw'+rpy_suffix] - phi_upwind_direction)
        kite_front_wrt_projected_kite_velocity[i] = calc_kite_front_wrt_projected_velocity(r_kite, v_kite, r_bw)
        v_app = v_kite - np.array([vwx, 0, 0])
        kite_front_wrt_projected_app_velocity[i] = calc_kite_front_wrt_projected_velocity(r_kite, v_app, r_bw)
    ax[1].plot(flight_data.time, kite_front_wrt_projected_kite_velocity*180./np.pi, label='kite velocity')
    ax[1].plot(flight_data.time, kite_front_wrt_projected_app_velocity*180./np.pi, label='apparent velocity')
    ax[1].plot(flight_data.time.iloc[mark_points], kite_front_wrt_projected_app_velocity[mark_points]*180./np.pi, 's')
    ax[1].set_ylabel('Kite front wrt\nprojected velocity [deg]')
    ax[1].set_xlabel('Time [s]')
    for a in ax: a.grid()


def plot_heading_and_course(flight_data):
    phi_upwind_direction = -flight_data.loc[flight_data.index[0], 'est_upwind_direction']-np.pi/2.
    heading = np.empty(flight_data.shape[0])
    course = np.empty(flight_data.shape[0])
    for i, (idx, row) in enumerate(flight_data.iterrows()):
        v_kite = np.array(list(row[['vx', 'vy', 'vz']]))
        r_bw = rotation_matrix_earth2body(row['roll'+rpy_suffix], row['pitch'+rpy_suffix], row['yaw'+rpy_suffix] - phi_upwind_direction)
        r_sw = rotation_matrix_earth2sphere(row['kite_azimuth'], row['kite_elevation'], 0)
        vector_s = r_sw.dot(r_bw.T[:, 0])
        heading[i] = np.arctan2(vector_s[1], vector_s[0])
        vector_s = r_sw.dot(v_kite)
        course[i] = np.arctan2(vector_s[1], vector_s[0])

    fig, ax = plt.subplots(3, 1)
    ax[0].plot(flight_data.time, heading*180./np.pi)
    ax[0].plot(flight_data.time, (np.pi - flight_data.kite_heading)*180./np.pi, '--')
    ax[0].plot(flight_data.time, flight_data.yaw0_tau * 180. / np.pi, ':', linewidth=3)
    ax[1].plot(flight_data.time, (heading - np.pi + flight_data.kite_heading)*180./np.pi, '--')
    ax[2].plot(flight_data.time, course*180./np.pi)
    ax[2].plot(flight_data.time, (np.pi - flight_data.kite_course)*180./np.pi, '--')


def animate_kite_attitude(flight_data, animate=False):
    from turning_center import mark_points
    ax = plt.figure().gca()

    phi_upwind_direction = -flight_data.loc[flight_data.index[0], 'est_upwind_direction']-np.pi/2.

    def plot(i, row=None):
        ax.cla()
        ax.set_title('Trajectory and kite\nlongitudinal axis projections')
        ax.set_xlim([-30, 30])
        ax.set_ylim([15, 55])
        ax.set_xlabel('Azimuth [deg]')
        ax.set_ylabel('Elevation [deg]')
        ax.set_aspect('equal')

        idx2i = flight_data.index[:i+1]
        ax.plot(flight_data.loc[idx2i, 'kite_azimuth']*180./np.pi, flight_data.loc[idx2i, 'kite_elevation']*180./np.pi)

        if row is not None and i < flight_data.shape[0]-1:
            plot_ex(ax, row, phi_upwind_direction)
            plot_apparent_wind_velocity(ax, row, vwx)

        for j in [mp for mp in mark_points if mp < i]:
            rowj = flight_data.iloc[j]
            plot_ex(ax, rowj, phi_upwind_direction, color='grey', linewidth=.5)
            plot_apparent_wind_velocity(ax, rowj, vwx, color='cyan', linewidth=.5)
            plot_kite_velocity(ax, rowj, color='blue', linewidth=.5)

    if animate:
        for i, (idx, row) in enumerate(flight_data.iterrows()):
            plot(i, row)
            plt.pause(0.001)
    else:
        plot(flight_data.shape[0])


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from utils import read_and_transform_flight_data

    flight_data = read_and_transform_flight_data()
    plot_velocity_and_respective_kite_attitude(flight_data)
    plot_heading_and_course(flight_data)
    animate_kite_attitude(flight_data)
    plt.show()
