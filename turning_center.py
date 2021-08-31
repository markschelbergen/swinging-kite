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
    flight_data['turn_radius'] = res[:, 2]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from utils import read_and_transform_flight_data
    flight_data = read_and_transform_flight_data()
    find_turns_for_rolling_window(flight_data)

    ax = plt.figure().gca()
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