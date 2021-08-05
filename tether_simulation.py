import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from generate_initial_state import get_moving_pyramid, get_tilted_pyramid_3d, check_constraints, \
    find_initial_velocities_satisfying_constraints
from tether_model_and_verification import derive_sim_input, dae_sim
from utils import calc_cartesian_coords_enu, plot_vector, unravel_euler_angles, plot_flight_sections, \
    read_and_transform_flight_data


def run_helical_flight():
    """"Imposing acceleration for turn with constant speed on last point mass to evaluate tether dynamics."""
    n_sections = 100
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
                                       n_sections, elevation_rotation_axis)
    r = np.zeros((n_sections, 3))
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

    dyn = derive_sim_input(n_sections, False, False, omega=omega, vwx=10)
    check_constraints(dyn, x0)
    run_simulation_and_plot_results(dyn, tf, n_intervals, x0, u)


def run_simulation_and_plot_results(dyn, tf, n_intervals, x0, u, animate=True, df=None):
    t = np.arange(0, n_intervals+1)*tf

    sim = dae_sim(tf, n_intervals, dyn)
    sol_x, sol_nu = sim(x0, u)
    sol_x = np.array(sol_x)
    sol_nu = np.array(sol_nu)

    rx = np.hstack((np.zeros((n_intervals+1, 1)), sol_x[:, :dyn['n_sections']*3:3]))
    ry = np.hstack((np.zeros((n_intervals+1, 1)), sol_x[:, 1:dyn['n_sections']*3:3]))
    rz = np.hstack((np.zeros((n_intervals+1, 1)), sol_x[:, 2:dyn['n_sections']*3:3]))

    tether_section_lengths = ((rx[:, 1:] - rx[:, :-1])**2 + (ry[:, 1:] - ry[:, :-1])**2 + (rz[:, 1:] - rz[:, :-1])**2)**.5
    tether_force = sol_nu*tether_section_lengths[1:, :]

    plt.figure()
    plt.plot(t[1:], tether_force[:, 0], label='sim')
    plt.xlabel("Time [s]")
    plt.ylabel("Tether force ground [N]")
    if df is not None:
        plt.plot(df.time, df.ground_tether_force, label='mea')
        plt.legend()
        plot_flight_sections(plt.gca(), df)

    get_rotation_matrices = ca.Function('get_rotation_matrices', [dyn['x'], dyn['u']],
                                        [dyn['rotation_matrices']['tangential_plane'],
                                         dyn['rotation_matrices']['end_section']])
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

    fig, ax_ypr = plt.subplots(3, 1, sharex=True)
    plt.suptitle("3-2-1 Euler angles between tangential\nand last tether element ref. frame")
    ax_ypr[0].plot(t[1:], ypr[:, 0]*180./np.pi, label='sim')
    ax_ypr[0].set_ylabel("Yaw [deg]")
    ax_ypr[1].plot(t[1:], ypr[:, 1]*180./np.pi, label='sim')
    ax_ypr[1].set_ylabel("Pitch [deg]")
    ax_ypr[2].plot(t[1:], ypr[:, 2]*180./np.pi, label='sim')
    ax_ypr[2].set_ylabel("Roll [deg]")
    ax_ypr[2].set_xlabel("Time [s]")
    if df is not None:
        ax_ypr[1].plot(df.time, df.pitch_tau*180./np.pi, label='mea')
        ax_ypr[2].plot(df.time, df.roll_tau*180./np.pi, label='mea')
        ax_ypr[2].legend()
        for a in ax_ypr:
            plot_flight_sections(a, df)

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
            # ax3d.plot3D([0, np.cos(elevation_rotation_axis)*150], [0, 0], [0, np.sin(elevation_rotation_axis)*150],
            #             '--', linewidth=.7, color='grey')  # Plot rotation axis
            ax3d.plot3D(rx[:, -1], ry[:, -1], rz[:, -1], linewidth=.7, color='grey')  # Plot trajectory of end point
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
        ax3d.plot3D(rx[:, -1], ry[:, -1], rz[:, -1], linewidth=.7, color='grey')  # Plot trajectory of end point

    fig, ax = plt.subplots(2, 1, sharey=True)
    plt.suptitle("Start and final position")
    ax[0].plot(rx[-1, :], rz[-1, :], 's-')
    ax[0].plot(rx[0, :], rz[0, :], '--')
    ax[0].set_xlim([0, 120])
    ax[0].set_xlabel("x [m]")
    ax[0].set_ylabel("z [m]")
    ax[0].axis('equal')

    ax[1].plot(ry[-1, :], rz[-1, :], 's-')
    ax[1].plot(ry[0, :], rz[0, :], '--')
    ax[1].set_xlabel("y [m]")
    ax[1].set_ylabel("z [m]")
    ax[1].axis('equal')


def run_simulation_with_measured_acceleration():
    df = read_and_transform_flight_data()
    x_kite, a_kite = find_accelerations_for_trajectory(df)
    r0, r1 = x_kite[0, :3], x_kite[1, :3]
    tf = .1
    n_intervals = a_kite.shape[0]

    n_sections = 30
    dyn = derive_sim_input(n_sections, False, False, vwx=9, impose_acceleration_directly=True)

    # Get starting position
    dl0 = 1.3
    l0 = np.sum(r0**2)**.5+1

    x0, y0, z0 = get_tilted_pyramid_3d(l0, *r0, n_sections)
    r = np.empty((n_sections, 3))
    r[:, 0] = x0
    r[:, 1] = y0
    r[:, 2] = z0

    x1, y1, z1 = get_tilted_pyramid_3d(l0+dl0*tf, *r1, n_sections)
    v = np.empty((n_sections, 3))
    v[:, 0] = (x1-x0)/tf
    v[:, 1] = (y1-y0)/tf
    v[:, 2] = (z1-z0)/tf

    x0 = np.vstack((r.reshape((-1, 1)), v.reshape((-1, 1)), [[l0], [dl0]]))
    x0 = find_initial_velocities_satisfying_constraints(dyn, x0, x_kite[0, 3:])

    check_constraints(dyn, x0)

    u = np.zeros((n_intervals, 4))
    u[:, 1:] = a_kite

    run_simulation_and_plot_results(dyn, tf, n_intervals, x0, u, animate=True, df=df)


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


def find_accelerations_for_trajectory(df, verify=False):
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


if __name__ == "__main__":
    # run_helical_flight()
    run_simulation_with_measured_acceleration()
    plt.show()
