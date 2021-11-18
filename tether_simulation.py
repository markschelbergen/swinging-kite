import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from generate_initial_state import get_moving_pyramid, get_tilted_pyramid_3d, check_constraints, \
    find_initial_velocities_satisfying_constraints
from tether_model_and_verification import derive_tether_model, derive_tether_model_kcu, derive_tether_model_kcu_williams, dae_sim
from utils import calc_cartesian_coords_enu, plot_vector, unravel_euler_angles, plot_flight_sections, \
    read_and_transform_flight_data, add_panel_labels
from paul_williams_model import get_tether_end_position, plot_offaxial_tether_displacement
from scipy.optimize import least_squares
from turning_center import determine_rigid_body_rotation, mark_points


def run_helical_flight():
    """"Imposing acceleration for turn with constant speed on last point mass to evaluate tether dynamics."""
    n_elements = 100
    elevation_rotation_axis = 30*np.pi/180.
    turning_radius = 25
    r0_kite = 100
    d_el = np.arcsin(turning_radius/r0_kite)
    # Start at lowest point
    start_position_kite = calc_cartesian_coords_enu(0, elevation_rotation_axis-d_el, r0_kite)
    min_tether_length = (start_position_kite[0]**2+start_position_kite[2]**2)**.5
    angular_speed = .8
    omega = calc_cartesian_coords_enu(0, elevation_rotation_axis, 1) * angular_speed
    omega = omega.reshape((1, 3))
    sim_periods = 2
    n_intervals = 100
    sim_time = 2*np.pi*sim_periods/angular_speed
    tf = sim_time/n_intervals  # Time step
    print("Simulation time and time step: {:.1f} s and {:.2f} s.".format(sim_time, tf))

    # Get starting position
    l0 = 100.5
    assert l0 > min_tether_length
    dl0 = 1

    v_rotation_center = dl0-.15

    assert start_position_kite[1] == 0
    x, z, vx, vz = get_moving_pyramid(l0, start_position_kite[0], start_position_kite[2], dl0, v_rotation_center,
                                       n_elements, elevation_rotation_axis)
    r = np.zeros((n_elements, 3))
    r[:, 0] = x
    r[:, 2] = z

    v = []
    for ri in r:
        v.append(np.cross(omega, ri))
    v = np.vstack(v)
    v[:, 0] = v[:, 0] + vx
    v[:, 2] = v[:, 2] + vz

    x0 = np.vstack((r.reshape((-1, 1)), v.reshape((-1, 1)), [[l0], [dl0]]))

    # Run simulation
    u = np.zeros((n_intervals, 1))

    dyn = derive_tether_model(n_elements, False, False, omega=omega, vwx=10)
    check_constraints(dyn, x0)
    run_simulation_and_plot_results(dyn, tf, n_intervals, x0, u)


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
    plt.plot(t[1:], tether_force[:, 0]*1e-3, label='Dyn')
    plt.ylabel("Tether force ground [kN]")
    plt.ylim([0, 5.5])
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

    # ax[1].plot(t, sol_x[:, -2]-r_end, label='sim')
    # ax[1].set_ylabel("Difference tether length\nand radial position [m]")
    # ax[1].set_ylim([0, None])
    # ax[1].plot(t, sol_x[:, -1], label='sim')
    # ax[1].set_ylabel("Tether speed [m/s]")
    # ax[1].set_ylim([0, None])
    # for a in ax: a.grid()
    # if flight_data is not None:
    #     ax[0].plot(flight_data.time, flight_data.ground_tether_force * 1e-3, label='mea')
    #     ax[1].plot(flight_data.time, flight_data.ground_tether_reelout_speed, label='mea')
    #     ax[1].legend()
    #     for a in ax: plot_flight_sections(a, flight_data)
    # add_panel_labels(ax)

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
        'pitch_bridle': ypr[:, 1],
        'roll_bridle': ypr[:, 2],
        'offaxial_tether_shape': pos_tau,
    }
    import pickle
    with open("dynamic_results{}.pickle".format(dyn['n_tether_elements']), 'wb') as f:
        pickle.dump(res, f)

    fig, ax_ypr = plt.subplots(3, 1, sharex=True)
    if plot_interval:
        plt.xlim([flight_data.loc[plot_interval_idx[0], 'time'], flight_data.loc[plot_interval_idx[1], 'time']])
    else:
        plt.xlim([flight_data.iloc[0]['time'], flight_data.iloc[-1]['time']])
    plt.suptitle("3-2-1 Euler angles between tangential\nand last tether element ref. frame")
    # for i in mark_points:
    #     ax_ypr[1].plot(t[i], ypr[i, 1]*180./np.pi, 's')
    #     ax_ypr[2].plot(t[i], ypr[i, 2]*180./np.pi, 's')
    # ax_ypr[0].plot(t, ypr[:, 0]*180./np.pi, label='sim')
    # ax_ypr[0].set_ylabel("Yaw [deg]")
    ax_ypr[0].plot(t, ypr[:, 1]*180./np.pi, label='sim')
    ax_ypr[0].set_ylabel("Pitch [deg]")
    ax_ypr[1].plot(t, ypr[:, 2]*180./np.pi, label='sim')
    ax_ypr[1].set_ylabel("Roll [deg]")
    ax_ypr[2].plot(t, ypr[:, 0]*180./np.pi, label='sim')
    ax_ypr[-1].set_xlabel("Time [s]")
    add_panel_labels(ax_ypr)

    for a in ax_ypr: a.grid()
    if flight_data is not None:
        # ypr_bridle_rbr = np.load('ypr_bridle_rigid_body_rotation.npy')
        ax_ypr[0].plot(flight_data.time, flight_data.pitch0_tau * 180. / np.pi, label='mea0')
        ax_ypr[0].plot(flight_data.time, flight_data.pitch1_tau * 180. / np.pi, label='mea1')
        # ax_ypr[0].plot(flight_data.time, ypr_bridle_rbr[:, 1] * 180. / np.pi, '--', label='rbr')
        ax_ypr[1].plot(flight_data.time, flight_data.roll0_tau * 180. / np.pi, label='mea0')
        ax_ypr[1].plot(flight_data.time, flight_data.roll1_tau * 180. / np.pi, label='mea1')
        # ax_ypr[1].plot(flight_data.time, ypr_bridle_rbr[:, 2] * 180. / np.pi, '--', label='rbr')
        ax_ypr[1].legend()
        for a in ax_ypr: plot_flight_sections(a, flight_data)

    plt.figure(figsize=(8, 6))
    ax3d = plt.axes(projection='3d')

    if animate:
        for i, ti in enumerate(t):
            ax3d.cla()

            ax3d.set_xlim([0, 250])
            ax3d.set_ylim([-125, 125])
            ax3d.set_zlim([0, 250])

            ax3d.plot3D(rx[i, :], ry[i, :], rz[i, :])
            for r in zip(rx[:i:25, :], ry[:i:25, :], rz[:i:25, :]):
                ax3d.plot3D(r[0], r[1], r[2], linewidth=.5, color='grey')
            # ax3d.plot3D([0, np.cos(elevation_rotation_axis)*150], [0, 0], [0, np.sin(elevation_rotation_axis)*150],
            #             '--', linewidth=.7, color='grey')  # Plot rotation axis
            ax3d.plot3D(rx[:, -1], ry[:, -1], rz[:, -1], color='grey')  # Plot trajectory of end point
            ax3d.text(75, 40, 75, "{:.2f} s".format(ti))

            ax3d.set_xlabel("x [m]")
            ax3d.set_ylabel("y [m]")
            ax3d.set_zlabel("z [m]")

            if i > 0:
                r = [rx[i, -1], ry[i, -1], rz[i, -1]]

                ex_tau = dcms_tau2e[i-1, :, 0]
                plot_vector(r, ex_tau, ax3d, scale_vector=15, color='k', label='tau ref frame')
                ey_tau = dcms_tau2e[i-1, :, 1]
                plot_vector(r, ey_tau, ax3d, scale_vector=15, color='k', label=None)

                ex_t = dcms_t2e[i-1, :, 0]
                plot_vector(r, ex_t, ax3d, scale_vector=15, color='g', label='tether end ref frame')
                ey_t = dcms_t2e[i-1, :, 1]
                plot_vector(r, ey_t, ax3d, scale_vector=15, color='g', label=None)
                plt.legend()

            plt.pause(0.001)
    else:
        # for r in zip(rx[::25, :], ry[::25, :], rz[::25, :]):
        #     ax3d.plot3D(r[0], r[1], r[2], linewidth=.7, color='grey')
        i0, i1 = plot_interval_irow
        for i in mark_points:
            ax3d.plot3D(rx[i+i0, :], ry[i+i0, :], rz[i+i0, :])
        ax3d.plot3D(rx[i0:i1, -1], ry[i0:i1, -1], rz[i0:i1, -1])  # Plot trajectory of end point

        ax3d.set_xlim([0, 250])
        ax3d.set_ylim([-125, 125])
        ax3d.set_zlim([0, 250])

        ax3d.set_xlabel("x [m]")
        ax3d.set_ylabel("y [m]")
        ax3d.set_zlabel("z [m]")

    # fig, ax = plt.subplots(2, 1, sharey=True)
    # plt.suptitle("Start and final position")
    # ax[0].plot(rx[-1, :], rz[-1, :], 's-')
    # ax[0].plot(rx[0, :], rz[0, :], '--')
    # ax[0].set_xlim([0, 120])
    # ax[0].set_xlabel("x [m]")
    # ax[0].set_ylabel("z [m]")
    # ax[0].axis('equal')
    #
    # ax[1].plot(ry[-1, :], rz[-1, :], 's-')
    # ax[1].plot(ry[0, :], rz[0, :], '--')
    # ax[1].set_xlabel("y [m]")
    # ax[1].set_ylabel("z [m]")
    # ax[1].axis('equal')

    return sol_x, sol_nu


def match_measured_tether_speed(df):
    n_intervals = df.shape[0]-1
    tf = .1

    # ODE for kinematic model
    dl = ca.SX.sym('dl')
    ddl = ca.SX.sym('ddl')

    # Create an integrator
    dae = {'x': dl, 'ode': ddl, 'p': ddl}

    intg = ca.integrator('intg', 'idas', dae, {'tf': tf})

    opti = ca.casadi.Opti()

    # Decision variables for states
    states = opti.variable(n_intervals+1)
    # Decision variables for control vector
    controls = opti.variable(n_intervals)

    # Gap-closing shooting constraints
    for k in range(n_intervals):
        res = intg(x0=states[k], p=controls[k])
        opti.subject_to(states[k+1] == res["xf"])

    # Initial guesses
    opti.set_initial(states, df['ground_tether_reelout_speed'].values)
    opti.set_initial(controls, np.diff(df['ground_tether_reelout_speed'].values)/.1)

    opti.minimize(ca.sumsqr(states - df['ground_tether_reelout_speed'].values))

    # solve optimization problem
    opti.solver('ipopt')

    sol = opti.solve()

    return sol.value(controls)


def run_simulation_with_fitted_acceleration(config=None, animate=False):
    if config is None:
        # i_cycle = None only looks at fo8
        config = {
            'i_cycle': None,
            'input_file_suffix': 'fo8',
            'sim_interval': None,
        }
    tether_states_file = 'tether_states_{}.npy'.format(config['input_file_suffix'])

    from system_properties import vwx
    # Get tether model.
    n_tether_elements = 30
    dyn = derive_tether_model_kcu_williams(n_tether_elements, False, vwx=vwx, impose_acceleration_directly=True)

    flight_data = read_and_transform_flight_data(True, config['i_cycle'], config['input_file_suffix'])  # Read flight data.
    if config['sim_interval'] is not None:
        flight_data = flight_data.iloc[config['sim_interval'][0]:config['sim_interval'][1]]
    else:
        config['sim_interval'] = (0, flight_data.shape[0])

    determine_rigid_body_rotation(flight_data)
    tf = .1  # Time step of the simulation - fixed by flight data time resolution.
    n_intervals = flight_data.shape[0] - 1  # Number of simulation steps - fixed by selected flight data interval.

    # ddl = match_measured_tether_speed(flight_data)
    ddl = np.load(tether_states_file)[config['sim_interval'][0]:config['sim_interval'][1]-1, 2]

    # Set control input array for simulation.
    u = np.zeros((n_intervals, 4))
    u[:, 0] = ddl
    u[:, 1:] = flight_data[['ax', 'ay', 'az']].values[:-1, :]

    # Get starting position of simulation
    # dl0 = flight_data.loc[flight_data.index[0], 'ground_tether_reelout_speed']
    # delta_l0 = 1.75  # Initial difference between the radial position of the kite and tether length.
    dl0 = np.load(tether_states_file)[config['sim_interval'][0], 1]
    l0 = np.load(tether_states_file)[config['sim_interval'][0], 0]

    positions = []
    for i in range(2):
        row = flight_data.iloc[i]
        args = (l0, n_tether_elements, list(row[['rx', 'ry', 'rz']]), list(row[['omx_opt', 'omy_opt', 'omz_opt']]),
                True, False)
        opt_res = least_squares(get_tether_end_position, list(row[['kite_elevation', 'kite_azimuth', 'kite_distance']]), args=args,
                                kwargs={'find_force': True}, verbose=0)
        positions.append(get_tether_end_position(opt_res.x, *args, return_values=True, find_force=True)[0][1:, :])
    r = positions[0]
    v = (positions[1]-positions[0])/tf

    x0 = np.vstack((r.reshape((-1, 1)), v.reshape((-1, 1)), [[l0], [dl0]]))
    x0 = find_initial_velocities_satisfying_constraints(dyn, x0, flight_data.iloc[0][['vx', 'vy', 'vz']])
    check_constraints(dyn, x0)

    # Run simulation.
    sol_x, sol_nu = run_simulation_and_plot_results(dyn, tf, n_intervals, x0, u, animate=animate, flight_data=flight_data)

    infer_aero_forces = False
    if infer_aero_forces:
        dyn_explicit = derive_tether_model_kcu(n_tether_elements, True, vwx=vwx, impose_acceleration_directly=True)
        fun_b = ca.Function('f_b', [dyn_explicit['x'], dyn_explicit['u']], [dyn_explicit['b']])
        dyn_f = derive_tether_model_kcu(n_tether_elements, False, vwx=vwx, impose_acceleration_directly=False)
        fun_mat = ca.Function('f_mat', [dyn_f['x'], dyn_f['u']], [dyn_f['a'], dyn_f['c']])

        aero_forces = []
        for x, ui, nu in zip(sol_x[1:, :], u, sol_nu):
            b = fun_b(x, ui)
            ddx = b[:dyn_explicit['n_elements']*3]

            a, c = fun_mat(x, [ui[0], 0, 0, 0])
            eps = np.array(a@b - c)  # print(np.sum(np.abs(eps) > 1e-9)) -
            f_aero = eps[(dyn_f['n_elements']-1)*3:dyn_f['n_elements']*3]
            aero_forces.append(np.linalg.norm(f_aero))

            # a, c = fun_mat(x, [ui[0], *f_aero])
            # eps = np.array(a@b - c)
            # print(np.sum(np.abs(eps) > 1e-9))

        plt.figure()
        plt.plot(aero_forces)


if __name__ == "__main__":
    # config = {
    #     'i_cycle': None,
    #     'input_file_suffix': 'fo8',
    #     'sim_interval': None,
    # }
    config = {
        'i_cycle': 65,
        'input_file_suffix': 'rugid',
        'sim_interval': (270, 513),
    }
    run_simulation_with_fitted_acceleration(config)
    plt.show()
