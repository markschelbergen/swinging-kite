import numpy as np
from scipy.optimize import least_squares


def circle_residuals(c, x, y):
    return np.sqrt((x-c[0])**2 + (y-c[1])**2) - c[2]


r_limit = .25
center_estimate = [(0, 0.7, .1), (0, 0.4, .1)]


def find_turn_specs(az, el):
    for j in range(2):
        res = least_squares(circle_residuals, center_estimate[j], args=(az, el), verbose=0)
        xc, yc, r = res.x
        if r < r_limit:
            return (xc, yc, r)
    return (np.nan, np.nan, np.nan)


def find_turns_for_rolling_window(flight_data):
    window = 9
    n_nans = (window-1)//2

    res = np.empty((flight_data.shape[0], 3))
    res[:] = np.nan
    for i in range(n_nans, flight_data.shape[0]-n_nans):
        if np.abs(flight_data.iloc[i].kite_azimuth) < .1:
            continue
        az = np.array(flight_data[i-n_nans:i+n_nans+1].kite_azimuth)
        el = np.array(flight_data[i-n_nans:i+n_nans+1].kite_elevation)
        res[i, :] = find_turn_specs(az, el)
    flight_data['azimuth_turn_center'] = res[:, 0]
    flight_data['elevation_turn_center'] = res[:, 1]
    # flight_data['turn_radius'] = res[:, 2]


def evaluate_acceleration(flight_data):
    from utils import plot_vector, plot_vector_2d, rotation_matrix_earth_sphere

    ax = plt.figure().gca()
    ax.set_aspect('equal')
    plt.plot(flight_data['kite_azimuth'], flight_data['kite_elevation'])

    counter = 0
    for idx, row in flight_data.iterrows():
        if not np.isnan(row['elevation_turn_center']):
            if counter % 15 == 0:
                # Project measured acceleration on sphere surface
                a_kite = row[['ax', 'ay', 'az']].values
                v_kite = row[['vx', 'vy', 'vz']].values

                if not np.isnan(row['elevation_turn_center']):
                    # Turning kite
                    r_kite = np.array(list(row[['rx', 'ry', 'rz']]))
                    elt, azt = row['elevation_turn_center'], row['azimuth_turn_center']
                    ax.plot([row['kite_azimuth'], azt], [row['kite_elevation'], elt], 's')
                    e_omega = np.array([np.cos(elt)*np.cos(azt), np.cos(elt)*np.sin(azt), np.sin(elt)])
                    r_parallel = np.dot(r_kite, e_omega)*e_omega
                    r_turn = r_kite - r_parallel

                    e_v = np.cross(e_omega, r_kite)
                    e_v = e_v/np.linalg.norm(e_v)
                    v_kite_rot = np.dot(v_kite, e_v)*e_v

                    om = np.cross(r_turn, v_kite_rot)/np.linalg.norm(r_turn)**2
                    a_kite_rot = np.cross(om, np.cross(om, r_kite))
                    plot_vectors = [a_kite, a_kite_rot]  #v_kite, v_kite_rot,
                else:
                    plot_vectors = [v_kite, a_kite]

                for vec, clr, ls in zip(plot_vectors, ['C1', 'C1', 'C2', 'C2'], ['-', ':', '-', ':']):
                    r_es = rotation_matrix_earth_sphere(row['kite_azimuth'], row['kite_elevation'], 0)
                    vec_sphere = r_es.T.dot(vec)
                    vec_proj = np.array([vec_sphere[1], -vec_sphere[0]])
                    plot_vector_2d([row['kite_azimuth'], row['kite_elevation']], vec_proj, ax, color=clr, linestyle=ls)

            counter += 1


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from utils import read_and_transform_flight_data

    animate = False
    flight_data = read_and_transform_flight_data()
    find_turns_for_rolling_window(flight_data)
    evaluate_acceleration(flight_data)

    ax = plt.figure().gca()
    if animate:
        for i in range(flight_data.shape[0]):
            ax.cla()
            ax.set_xlim([-.4, .4])
            ax.set_ylim([.4, .8])
            ax.set_xlabel('Azimuth [rad]')
            ax.set_ylabel('Elevation [rad]')
            ax.set_aspect('equal')

            ax.plot(flight_data.iloc[:i]['kite_azimuth'], flight_data.iloc[:i]['kite_elevation'])
            ax.plot(flight_data.iloc[:i]['azimuth_turn_center'], flight_data.iloc[:i]['elevation_turn_center'], linewidth=.5, color='grey')
            ax.plot(flight_data.iloc[i]['azimuth_turn_center'], flight_data.iloc[i]['elevation_turn_center'], 's')
            plt.pause(0.001)
    else:
        ax.set_xlim([-.4, .4])
        ax.set_ylim([.4, .8])
        ax.set_xlabel('Azimuth [rad]')
        ax.set_ylabel('Elevation [rad]')
        ax.set_aspect('equal')

        ax.plot(flight_data['kite_azimuth'], flight_data['kite_elevation'])
        ax.plot(flight_data['azimuth_turn_center'], flight_data['elevation_turn_center'], linewidth=.5, color='grey')
    plt.show()