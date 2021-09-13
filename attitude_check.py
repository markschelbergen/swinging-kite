import numpy as np


def calc_kite_kite_front_wrt_projected_velocity(r_kite, v_kite, rm_earth2body):
    ez_v = r_kite/np.linalg.norm(r_kite)
    ey_v = np.cross(ez_v, v_kite)/np.linalg.norm(np.cross(ez_v, v_kite))
    ex_v = np.cross(ey_v, ez_v)
    rm_v2e_i = np.vstack(([ex_v], [ey_v], [ez_v])).T

    exb_earth = rm_earth2body.T[:, 0]
    exb_v = rm_v2e_i.T.dot(exb_earth)
    return np.arctan2(exb_v[1], exb_v[0])


def plot_velocity_and_respective_kite_attitude(flight_data):
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(flight_data.time, np.sum(flight_data[['vx', 'vy', 'vz']]**2, axis=1)**.5)
    ax[0].set_ylabel('Kite speed [m/s]')

    kite_front_wrt_projected_velocity = np.empty(flight_data.shape[0])
    for i, (idx, row) in enumerate(flight_data.iterrows()):
        r_kite = np.array(list(row[['rx', 'ry', 'rz']]))
        v_kite = np.array(list(row[['vx', 'vy', 'vz']]))
        rm_earth2body = rotation_matrix_earth2body(row['roll'], row['pitch'], row['yaw'])
        kite_front_wrt_projected_velocity[i] = calc_kite_kite_front_wrt_projected_velocity(r_kite, v_kite, rm_earth2body)
    ax[1].plot(flight_data.time, kite_front_wrt_projected_velocity*180./np.pi)
    ax[1].set_ylabel('Kite front wrt\nprojected velocity [deg]')
    ax[1].set_xlabel('Time [s]')
    for a in ax: a.grid()


def animate_kite_attitude(flight_data, vwx=10, animate=True):
    ax = plt.figure().gca()
    if animate:
        for i, (idx, row) in enumerate(flight_data.iterrows()):
            ax.cla()
            ax.set_title('Trajectory and kite\nlongitudinal axis projections')
            ax.set_xlim([-30, 30])
            ax.set_ylim([15, 55])
            ax.set_xlabel('Azimuth [deg]')
            ax.set_ylabel('Elevation [deg]')
            ax.set_aspect('equal')

            idx2i = flight_data.index[:i+1]
            ax.plot(flight_data.loc[idx2i, 'kite_azimuth']*180./np.pi, flight_data.loc[idx2i, 'kite_elevation']*180./np.pi)

            def plot_ex(row, **kwargs):
                r_es = rotation_matrix_earth_sphere(row['kite_azimuth'], row['kite_elevation'], 0)
                rm_earth2body = rotation_matrix_earth2body(row['roll'], row['pitch'], row['yaw'])
                exb_earth = rm_earth2body.T[:, 0]

                vec_sphere = r_es.T.dot(exb_earth)
                vec_proj = np.array([vec_sphere[1], -vec_sphere[0]])
                plot_vector_2d([row['kite_azimuth']*180./np.pi, row['kite_elevation']*180./np.pi], vec_proj, ax, 1e1, **kwargs)

            def plot_apparent_wind_velocity(row, **kwargs):
                v_kite = np.array(list(row[['vx', 'vy', 'vz']]))
                v_app = v_kite - np.array([vwx, 0, 0])
                r_es = rotation_matrix_earth_sphere(row['kite_azimuth'], row['kite_elevation'], 0)

                vec_sphere = r_es.T.dot(v_app)
                vec_proj = np.array([vec_sphere[1], -vec_sphere[0]])
                plot_vector_2d([row['kite_azimuth']*180./np.pi, row['kite_elevation']*180./np.pi], vec_proj, ax, 1, **kwargs)


            plot_ex(row)
            plot_apparent_wind_velocity(row)

            for j in range(0, i, 10):
                rowj = flight_data.iloc[j]
                plot_ex(rowj, color='grey', linewidth=.5)
                plot_apparent_wind_velocity(rowj, color='cyan', linewidth=.5)

            plt.pause(0.001)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from utils import read_and_transform_flight_data, rotation_matrix_earth_sphere, plot_vector_2d, rotation_matrix_earth2body

    flight_data = read_and_transform_flight_data()
    plot_velocity_and_respective_kite_attitude(flight_data)
    animate_kite_attitude(flight_data)
    plt.show()