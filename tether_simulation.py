import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from generate_initial_state import get_moving_pyramid, get_tilted_pyramid_3d, check_constraints, \
    find_initial_velocities_satisfying_constraints
from tether_model_and_verification import derive_tether_model, derive_tether_model_kcu, dae_sim, l_bridle
from utils import calc_cartesian_coords_enu, plot_vector, unravel_euler_angles, plot_flight_sections, \
    read_and_transform_flight_data


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


def run_simulation_and_plot_results(dyn, tf, n_intervals, x0, u, animate=True, flight_data=None):
    t = np.arange(0, n_intervals+1)*tf

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

    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(t[1:], tether_force[:, 0]*1e-3, label='sim')
    ax[0].set_ylabel("Tether force ground [kN]")
    ax[0].set_ylim([0, 5.5])
    ax[1].plot(t, sol_x[:, -2]-r_end, label='sim')
    ax[1].set_ylabel("Difference tether length\nand radial position [m]")
    ax[1].set_ylim([0, None])
    ax[2].plot(t, sol_x[:, -1], label='sim')
    ax[2].set_ylabel("Tether speed [m/s]")
    ax[2].set_xlabel("Time [s]")
    ax[2].set_ylim([0, None])
    for a in ax: a.grid()
    if flight_data is not None:
        ax[0].plot(flight_data.time, flight_data.ground_tether_force * 1e-3, label='mea')
        ax[2].plot(flight_data.time, flight_data.ground_tether_reelout_speed, label='mea')
        ax[2].legend()
        for a in ax: plot_flight_sections(a, flight_data)

    get_rotation_matrices = ca.Function('get_rotation_matrices', [dyn['x'], dyn['u']],
                                        [dyn['rotation_matrices']['tangential_plane'],
                                         dyn['rotation_matrices']['last_element']])
    ypr = np.empty((n_intervals, 3))
    rm_tau2e = np.empty((n_intervals, 3, 3))
    rm_t2e = np.empty((n_intervals, 3, 3))
    for i, (xi, ui) in enumerate(zip(sol_x[1:, :], u)):  # Determine at end of interval - as done for lagrangian multipliers
        dcms = get_rotation_matrices(xi, ui)
        rm_tau2e_i = np.array(dcms[0])
        rm_tau2e[i, :, :] = rm_tau2e_i
        rm_t2e_i = np.array(dcms[1])
        rm_t2e[i, :, :] = rm_t2e_i
        rm_tau2t = rm_t2e_i.T.dot(rm_tau2e_i)

        ypr[i, :] = unravel_euler_angles(rm_tau2t, '321')

    highlight_time_points = [0, 97, 107, 205, 215]

    fig, ax_ypr = plt.subplots(3, 1, sharex=True)
    plt.suptitle("3-2-1 Euler angles between tangential\nand last tether element ref. frame")
    ax_ypr[0].plot(t[1:], ypr[:, 0]*180./np.pi, label='sim')
    ax_ypr[0].set_ylabel("Yaw [deg]")
    ax_ypr[1].plot(t[1:], ypr[:, 1]*180./np.pi, label='sim')
    ax_ypr[1].set_ylabel("Pitch [deg]")
    ax_ypr[2].plot(t[1:], ypr[:, 2]*180./np.pi, label='sim')
    ax_ypr[2].set_ylabel("Roll [deg]")
    ax_ypr[2].set_xlabel("Time [s]")
    for i in highlight_time_points:
        ax_ypr[1].plot(t[i], ypr[i+1, 1]*180./np.pi, 's')
        ax_ypr[2].plot(t[i], ypr[i+1, 2]*180./np.pi, 's')

    for a in ax_ypr: a.grid()
    if flight_data is not None:
        ax_ypr[1].plot(flight_data.time, flight_data.pitch_tau * 180. / np.pi, label='mea')
        ax_ypr[2].plot(flight_data.time, flight_data.roll_tau * 180. / np.pi, label='mea')
        ax_ypr[2].legend()
        for a in ax_ypr: plot_flight_sections(a, flight_data)

    plt.figure(figsize=(8, 6))
    ax3d = plt.axes(projection='3d')

    if animate:
        for i, ti in enumerate(t):
            ax3d.cla()
            # ax3d.set_xlim([0, 150])
            # ax3d.set_ylim([-75, 75])
            # ax3d.set_zlim([0, 150])

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

                ex_tau = rm_tau2e[i-1, :, 0]
                plot_vector(r, ex_tau, ax3d, scale_vector=15, color='k', label='tau ref frame')
                ey_tau = rm_tau2e[i-1, :, 1]
                plot_vector(r, ey_tau, ax3d, scale_vector=15, color='k', label=None)

                ex_t = rm_t2e[i-1, :, 0]
                plot_vector(r, ex_t, ax3d, scale_vector=15, color='g', label='tether end ref frame')
                ey_t = rm_t2e[i-1, :, 1]
                plot_vector(r, ey_t, ax3d, scale_vector=15, color='g', label=None)
                plt.legend()

            plt.pause(0.001)
    else:
        # for r in zip(rx[::25, :], ry[::25, :], rz[::25, :]):
        #     ax3d.plot3D(r[0], r[1], r[2], linewidth=.7, color='grey')
        ax3d.plot3D(rx[:, -1], ry[:, -1], rz[:, -1])  # Plot trajectory of end point
        for i in highlight_time_points:
            ax3d.plot3D(rx[i, :], ry[i, :], rz[i, :])

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


def setup_integrator_kinematic_model(tf):
    # ODE for kinematic model
    r = ca.SX.sym('r', 3)
    v = ca.SX.sym('v', 3)
    x = ca.vertcat(r, v)
    a = ca.SX.sym('a', 3)
    dx = ca.vertcat(v, a)

    # Create an integrator
    dae = {'x': x, 'ode': dx, 'p': a}

    intg = ca.integrator('intg', 'idas', dae, {'tf': tf})
    return intg


def find_acceleration_matching_kite_trajectory(df, verify=False):
    n_intervals = df.shape[0]-1
    tf = .1

    intg = setup_integrator_kinematic_model(tf)

    opti = ca.casadi.Opti()

    # Decision variables for states
    states = opti.variable(n_intervals+1, 6)
    # Decision variables for control vector
    controls = opti.variable(n_intervals, 3)

    # Gap-closing shooting constraints
    for k in range(n_intervals):
        res = intg(x0=states[k, :], p=controls[k, :])
        opti.subject_to(states[k+1, :].T == res["xf"])

    # Initial guesses
    opti.set_initial(states, df[['rx', 'ry', 'rz', 'vx', 'vy', 'vz']].values)
    opti.set_initial(controls, df.loc[df.index[:-1], ['ax', 'ay', 'az']].values)

    opti.minimize(ca.sumsqr(states - df[['rx', 'ry', 'rz', 'vx', 'vy', 'vz']].values))

    # solve optimization problem
    opti.solver('ipopt')

    sol = opti.solve()

    states_sol = sol.value(states)
    controls_sol = sol.value(controls)

    # # Uncomment lower to evaluate results using measurements directly
    # states_sol[0, :] = df.loc[df.index[0], ['rx', 'ry', 'rz', 'kite_1_vy', 'kite_1_vx', 'kite_1_vz']]
    # states_sol[0, -1] = -states_sol[0, -1]
    # controls_sol = np.vstack(([df['kite_1_ay'].values], [df['kite_1_ax'].values], [-df['kite_1_az'].values])).T  #

    if verify:
        x_sol = [states_sol[0, :]]
        for i in range(n_intervals):
            sol = intg(x0=x_sol[-1], p=controls_sol[i, :])
            x_sol.append(sol["xf"].T)
        x_sol = np.vstack(x_sol)

        plt.figure(figsize=(8, 6))
        ax3d = plt.axes(projection='3d')
        ax3d.set_xlim([0, 250])
        ax3d.set_ylim([-125, 125])
        ax3d.set_zlim([0, 250])
        ax3d.plot3D(x_sol[:, 0], x_sol[:, 1], x_sol[:, 2])
        ax3d.plot3D(df['rx'], df['ry'], df['rz'], '--')

    return states_sol, controls_sol


def find_matching_tether_acceleration(df):
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


def run_simulation_with_measured_acceleration(realistic_tether_input=False):
    # Get tether model.
    separate_kcu_mass = True
    n_tether_elements = 30
    dyn = derive_tether_model_kcu(n_tether_elements, separate_kcu_mass, False, vwx=9, impose_acceleration_directly=True)

    flight_data = read_and_transform_flight_data()  # Read flight data.
    # for k in list(df): print(k)
    tf = .1  # Time step of the simulation - fixed by flight data time resolution.
    n_intervals = flight_data.shape[0] - 1  # Number of simulation steps - fixed by selected flight data interval.

    # Control input for this simulation exists of tether acceleration and accelerations on last point mass.
    if realistic_tether_input:  # Infer tether acceleration from measurements.
        ddl = find_matching_tether_acceleration(flight_data)
    else:  # Run simulation with constant tether acceleration and, in case the latter is set to zero, constant tether
        # speed.
        ddl = 0

    # Infer kite acceleration from measurements.
    x_kite, a_kite = find_acceleration_matching_kite_trajectory(flight_data)

    # Set control input array for simulation.
    u = np.zeros((n_intervals, 4))
    u[:, 0] = ddl
    u[:, 1:] = a_kite

    # Get starting position of simulation
    if realistic_tether_input:
        dl0 = flight_data.loc[flight_data.index[0], 'ground_tether_reelout_speed']
        delta_l0 = 2  # Initial difference between the radial position of the kite and tether length.
    else:
        dl0 = 1.24
        delta_l0 = 1.2  #1.05  # Initial difference between the radial position of the kite and tether length.
    p0, p1 = x_kite[0, :3], x_kite[1, :3]
    r0 = np.sum(p0**2)**.5
    l0 = r0+delta_l0  # Lower to increase tether force, but setting too low results in the tether being
    # shorter than the radial position of the kite and thus a crashing simulation.
    if separate_kcu_mass:
        l0 -= l_bridle
        p0 = p0/np.linalg.norm(p0)*(r0-l_bridle)
        r1 = np.sum(p1**2)**.5
        p1 = p1/np.linalg.norm(p1)*(r1-l_bridle)

    x0, y0, z0 = get_tilted_pyramid_3d(l0, *p0, n_tether_elements)
    r = np.empty((n_tether_elements, 3))
    r[:, 0] = x0
    r[:, 1] = y0
    r[:, 2] = z0
    if separate_kcu_mass:
        r = np.vstack((r, x_kite[:1, :3]))

    x1, y1, z1 = get_tilted_pyramid_3d(l0+dl0*tf, *p1, n_tether_elements)
    v = np.empty((n_tether_elements, 3))
    v[:, 0] = (x1-x0)/tf
    v[:, 1] = (y1-y0)/tf
    v[:, 2] = (z1-z0)/tf
    if separate_kcu_mass:
        v = np.vstack((v, x_kite[:1, 3:6]))

    x0 = np.vstack((r.reshape((-1, 1)), v.reshape((-1, 1)), [[l0], [dl0]]))
    x0 = find_initial_velocities_satisfying_constraints(dyn, x0, x_kite[0, 3:])
    check_constraints(dyn, x0)

    # Run simulation.
    sol_x, sol_nu = run_simulation_and_plot_results(dyn, tf, n_intervals, x0, u, animate=True, flight_data=flight_data)

    # dyn_explicit = derive_tether_model_kcu(n_tether_elements, separate_kcu_mass, True, vwx=9, impose_acceleration_directly=True)
    # fun_b = ca.Function('f_b', [dyn_explicit['x'], dyn_explicit['u']], [dyn_explicit['b']])
    # dyn_f = derive_tether_model_kcu(n_tether_elements, separate_kcu_mass, False, vwx=9, impose_acceleration_directly=False)
    # fun_mat = ca.Function('f_mat', [dyn_f['x'], dyn_f['u']], [dyn_f['a'], dyn_f['c']])
    #
    # aero_forces = []
    # for x, ui, nu in zip(sol_x[1:, :], u, sol_nu):
    #     b = fun_b(x, ui)
    #     ddx = b[:dyn_explicit['n_elements']*3]
    #
    #     a, c = fun_mat(x, [ui[0], 0, 0, 0])
    #     eps = np.array(a@b - c)  # print(np.sum(np.abs(eps) > 1e-9)) -
    #     f_aero = eps[(dyn_f['n_elements']-1)*3:dyn_f['n_elements']*3]
    #     aero_forces.append(np.linalg.norm(f_aero))
    #
    #     # a, c = fun_mat(x, [ui[0], *f_aero])
    #     # eps = np.array(a@b - c)
    #     # print(np.sum(np.abs(eps) > 1e-9))
    #
    # plt.figure()
    # plt.plot(aero_forces)

if __name__ == "__main__":
    # run_helical_flight()
    run_simulation_with_measured_acceleration()
    plt.show()
