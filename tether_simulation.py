import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from generate_initial_state import get_moving_pyramid
from multi_element_tether import derive_sim_input, check_constraints, dae_sim
from utils import calc_cartesian_coords_enu, plot_vector, unravel_euler_angles


def run_helical_flight():
    """"Imposing acceleration for turn with constant speed on last point mass to evaluate tether dynamics."""
    animate = True
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
    t = np.arange(0, n_intervals+1)*tf

    # Get starting position
    l0 = 100.5
    assert l0 > min_tether_length
    dl0 = 1

    v_rotation_center = dl0-.15

    assert start_position_kite[1] == 0
    x, z, vx, vz = get_moving_pyramid(l0, start_position_kite[0], start_position_kite[2], dl0, v_rotation_center,
                                       n_sections, elevation_rotation_axis)
    n_free_pm = n_sections-1

    r = np.zeros((n_free_pm+1, 3))
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

    fix_end_in_derivation = False
    dyn = derive_sim_input(n_sections, fix_end_in_derivation, False, omega=omega, vwx=10)
    check_constraints(dyn, x0)

    sim = dae_sim(tf, n_intervals, dyn)
    sol_x, sol_nu = sim(x0, u)
    sol_x = np.array(sol_x)
    sol_nu = np.array(sol_nu)

    rx = np.hstack((np.zeros((n_intervals+1, 1)), sol_x[:, :n_sections*3:3]))
    ry = np.hstack((np.zeros((n_intervals+1, 1)), sol_x[:, 1:n_sections*3:3]))
    rz = np.hstack((np.zeros((n_intervals+1, 1)), sol_x[:, 2:n_sections*3:3]))

    tether_section_lengths = ((rx[:, 1:] - rx[:, :-1])**2 + (ry[:, 1:] - ry[:, :-1])**2 + (rz[:, 1:] - rz[:, :-1])**2)**.5
    tether_force = sol_nu*tether_section_lengths[1:, :]

    plt.figure()
    plt.plot(t[1:], tether_force[:, 0])
    plt.xlabel("Time [s]")
    plt.ylabel("Tether force ground [N]")

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

    fig, ax = plt.subplots(3, 1, sharex=True)
    plt.suptitle("3-2-1 Euler angles between tangential\nand last tether element ref. frame")
    ax[0].plot(t[1:], ypr[:, 0]*180./np.pi)
    ax[0].set_ylabel("Yaw [deg]")
    ax[1].plot(t[1:], ypr[:, 1]*180./np.pi)
    ax[1].set_ylabel("Pitch [deg]")
    ax[2].plot(t[1:], ypr[:, 2]*180./np.pi)
    ax[2].set_ylabel("Roll [deg]")
    ax[2].set_xlabel("Time [s]")

    if animate:
        plt.figure(figsize=(12, 12))
        ax3d = plt.axes(projection='3d')

        for i, ti in enumerate(t):
            ax3d.cla()
            ax3d.set_xlim([0, 150])
            ax3d.set_ylim([-75, 75])
            ax3d.set_zlim([0, 150])
            ax3d.plot3D(rx[i, :], ry[i, :], rz[i, :])
            ax3d.plot3D([0, np.cos(elevation_rotation_axis)*150], [0, 0], [0, np.sin(elevation_rotation_axis)*150],
                        '--', linewidth=.7, color='grey')  # Plot rotation axis
            ax3d.plot3D(rx[:, -1], ry[:, -1], rz[:, -1], linewidth=.7, color='grey')  # Plot trajectory of end point
            ax3d.text(75, 40, 75, "{:.2f} s".format(ti))

            if i > 0:
                r = [rx[i, -1], ry[i, -1], rz[i, -1]]

                ex_tau = rm_tau2e[i-1, :, 0]
                plot_vector(r, ex_tau, ax3d, scale_vector=turning_radius/2, color='k', label='tau ref frame')
                ey_tau = rm_tau2e[i-1, :, 1]
                plot_vector(r, ey_tau, ax3d, scale_vector=turning_radius/2, color='k', label=None)

                ex_t = rm_t2e[i-1, :, 0]
                plot_vector(r, ex_t, ax3d, scale_vector=turning_radius/2, color='g', label='tether end ref frame')
                ey_t = rm_t2e[i-1, :, 1]
                plot_vector(r, ey_t, ax3d, scale_vector=turning_radius/2, color='g', label=None)
                plt.legend()

            plt.pause(0.001)

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
    ax[1].axis('equal')

    plt.show()


if __name__ == "__main__":
    run_helical_flight()
    plt.show()
