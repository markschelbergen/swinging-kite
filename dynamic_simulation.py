import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from generate_initial_state import check_constraints, find_initial_velocities_satisfying_constraints
from dynamic_model import derive_tether_model_kcu_williams_wo_wing, dae_sim
from utils import plot_vector, unravel_euler_angles, plot_flight_sections, \
    read_and_transform_flight_data, add_panel_labels
from steady_rotation_routine import get_tether_end_position
from scipy.optimize import least_squares
from turning_center import determine_rigid_body_rotation, mark_points
from system_properties import l_bridle


def run_simulation_and_plot_results(dyn, tf, n_intervals, x0, u, animate=True, flight_data=None, plot_interval=(29.9, 51.2)):
    if flight_data is not None:
        t = flight_data.time
        plot_interval_idx = (
            (flight_data['time'] == plot_interval[0]).idxmax(),
            (flight_data['time'] == plot_interval[1]).idxmax()
        )
        plot_interval_irow = (
            plot_interval_idx[0] - flight_data.index[0],
            plot_interval_idx[1] - flight_data.index[0],
        )
    else:
        t = np.arange(0, n_intervals+1)*tf
        plot_interval_idx = False

    sim = dae_sim(tf, n_intervals, dyn)
    sol_x, sol_nu = sim(x0, u)
    sol_x = np.array(sol_x)
    sol_nu = np.array(sol_nu)

    rx = np.hstack((np.zeros((n_intervals+1, 1)), sol_x[:, :dyn['n_elements']*3:3]))
    ry = np.hstack((np.zeros((n_intervals+1, 1)), sol_x[:, 1:dyn['n_elements']*3:3]))
    rz = np.hstack((np.zeros((n_intervals+1, 1)), sol_x[:, 2:dyn['n_elements']*3:3]))

    tether_element_lengths = ((rx[:, 1:] - rx[:, :-1])**2 + (ry[:, 1:] - ry[:, :-1])**2 + (rz[:, 1:] - rz[:, :-1])**2)**.5
    tether_force = sol_nu*tether_element_lengths[1:, :]

    n_tether_elements = dyn.get('n_tether_elements', dyn['n_elements'])
    r_end = np.sum(sol_x[:, (n_tether_elements-1)*3:n_tether_elements*3]**2, axis=1)**.5

    # fig, ax = plt.subplots(2, 1, sharex=True)
    plt.figure(figsize=[4.8, 2.4])
    plt.subplots_adjust(top=0.97, bottom=0.2, left=0.1, right=0.985)
    plt.plot(t[1:], tether_force[:, 0]*1e-3, label='Dynamic model')
    plt.ylabel("Tether force ground [kN]")
    plt.ylim([0, None])
    plt.xlabel("Time [s]")
    plt.grid()
    if plot_interval:
        plt.xlim([flight_data.loc[plot_interval_idx[0], 'time'], flight_data.loc[plot_interval_idx[1], 'time']])
    else:
        plt.xlim([flight_data.iloc[0]['time'], flight_data.iloc[-1]['time']])
    if flight_data is not None:
        plt.plot(flight_data.time, flight_data.ground_tether_force * 1e-3, label='Measured')
        plt.legend()
        plot_flight_sections(plt.gca(), flight_data)

    get_rotation_matrices = ca.Function('get_rotation_matrices', [dyn['x'], dyn['u']],
                                        [dyn['rotation_matrices']['tangential_plane'],
                                         dyn['rotation_matrices']['last_element']])

    ypr = np.empty((n_intervals+1, 3))
    ypr[0, :] = np.nan
    dcms_tau2e = np.empty((n_intervals, 3, 3))
    dcms_t2e = np.empty((n_intervals, 3, 3))
    pos_tau = np.zeros((n_intervals+1, dyn['n_elements']+1, 3))
    pos_tau[0, :] = np.nan
    for i, (xi, ui) in enumerate(zip(sol_x[1:, :], u)):  # Determine at end of interval - as done for lagrangian multipliers
        dcms = get_rotation_matrices(xi, ui)
        dcm_tau2e_i = np.array(dcms[0])
        dcms_tau2e[i, :, :] = dcm_tau2e_i
        dcm_t2e_i = np.array(dcms[1])
        dcms_t2e[i, :, :] = dcm_t2e_i
        dcm_tau2t = dcm_t2e_i.T.dot(dcm_tau2e_i)

        ypr[i+1, :] = unravel_euler_angles(dcm_tau2t, '321')

        for j in range(dyn['n_elements']+1):
            pos_e = np.array((rx[i+1, j], ry[i+1, j], rz[i+1, j]))
            pos_tau[i+1, j, :] = dcm_tau2e_i.T.dot(pos_e)

    res = {
        'pitch_bridle': ypr[plot_interval_irow[0]:plot_interval_irow[1]+1, 1],
        'roll_bridle': ypr[plot_interval_irow[0]:plot_interval_irow[1]+1, 2],
        'offaxial_tether_shape': pos_tau[plot_interval_irow[0]:plot_interval_irow[1]+1],
    }
    import pickle
    with open("results/dynamic_results{}.pickle".format(dyn['n_tether_elements']), 'wb') as f:
        pickle.dump(res, f)

    return sol_x, sol_nu


def run_simulation_with_fitted_acceleration(config=None, animate=False):
    if config is None:
        # i_cycle = None only looks at fo8
        config = {
            'i_cycle': None,
            # 'input_file_suffix': 'fo8',
            'sim_interval': None,
            'tether_slack0': .3,
            'use_measured_reelout_acceleration': False,
        }
    from system_properties import vwx
    # Get tether model.
    n_tether_elements = 30
    # dyn = derive_tether_model_kcu_williams(n_tether_elements, False, vwx=vwx, impose_acceleration_directly=True)
    dyn = derive_tether_model_kcu_williams_wo_wing(n_tether_elements, vwx=vwx)

    flight_data = read_and_transform_flight_data(True, config['i_cycle'])
    if config['sim_interval'] is not None:
        flight_data = flight_data.iloc[config['sim_interval'][0]:config['sim_interval'][1]]
    else:
        config['sim_interval'] = (0, flight_data.shape[0])

    determine_rigid_body_rotation(flight_data)
    tf = .1  # Time step of the simulation - fixed by flight data time resolution.
    n_intervals = flight_data.shape[0] - 1  # Number of simulation steps - fixed by selected flight data interval.

    if config['use_measured_reelout_acceleration']:
        ddl = np.diff(flight_data['ground_tether_reelout_speed'].values)/.1
        dl0 = flight_data.loc[flight_data.index[0], 'ground_tether_reelout_speed']
        dl0 += .02
    else:
        ddl = flight_data['ddl'].iloc[:-1]
        dl0 = flight_data['dl'].iloc[0]
    l0 = flight_data['l'].iloc[0] - l_bridle + config['tether_slack0']

    # Set control input array for simulation.
    u = np.zeros((n_intervals, 4))
    u[:, 0] = ddl
    u[:, 1:] = flight_data[['ax', 'ay', 'az']].values[:-1, :]

    positions = []
    for i in range(2):
        row = flight_data.iloc[i]
        gtep_config = {
            'set_parameter': l0,
            'n_tether_elements': n_tether_elements,
            'r_kite': list(row[['rx', 'ry', 'rz']]),
            'omega': list(row[['omx_opt', 'omy_opt', 'omz_opt']]),
            'separate_kcu_mass': True,
            'elastic_elements': False,
            'find_force': True,
        }
        opt_res = least_squares(get_tether_end_position, list(row[['kite_elevation', 'kite_azimuth', 'kite_distance']]),
                                args=(gtep_config, ), verbose=0)
        gtep_config['return_values'] = True
        positions.append(get_tether_end_position(opt_res.x, gtep_config)[0][1:, :])
    r = positions[0]
    v = (positions[1]-positions[0])/tf

    x0 = np.vstack((r.reshape((-1, 1)), v.reshape((-1, 1)), [[l0], [dl0]]))
    x0 = find_initial_velocities_satisfying_constraints(dyn, x0, flight_data.iloc[0][['vx', 'vy', 'vz']])
    check_constraints(dyn, x0)

    # Run simulation.
    sol_x, sol_nu = run_simulation_and_plot_results(dyn, tf, n_intervals, x0, u, animate=animate, flight_data=flight_data)


if __name__ == "__main__":
    config = {
        'i_cycle': 65,
        'sim_interval': (270, 513),
        'tether_slack0': .3,
        'use_measured_reelout_acceleration': False,
    }
    run_simulation_with_fitted_acceleration(config)  # Generates dynamic results and plots figure 10
    plt.show()
