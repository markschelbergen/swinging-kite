import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import pickle

from utils import unravel_euler_angles, plot_flight_sections, rotation_matrix_earth2body, \
    read_and_transform_flight_data, add_panel_labels, get_pitch_nose_down_angle_v3
from system_properties import *
from tether_model_and_verification import derive_tether_model_kcu_williams
from attitude_check import calc_kite_front_wrt_projected_velocity
from turning_center import mark_points, determine_rigid_body_rotation


def plot_vector(p0, v, ax, scale_vector=.03, color='g', label=None):
    p1 = p0 + v * scale_vector
    vector = np.vstack(([p0], [p1])).T
    ax.plot3D(vector[0], vector[1], vector[2], color=color, label=label)


def get_tether_end_position(x, set_parameter, n_tether_elements, r_kite, omega, separate_kcu_mass=False, elastic_elements=True, ax_plot_forces=False, return_values=False, find_force=False):
    # Currently neglecting radial velocity of kite.
    if find_force:
        beta_n, phi_n, tension_ground = x
        tether_length = set_parameter
    else:
        beta_n, phi_n, tether_length = x
        tension_ground = set_parameter

    l_unstrained = tether_length/n_tether_elements
    m_s = np.pi*d_t**2/4 * l_unstrained * rho_t

    n_elements = n_tether_elements
    if separate_kcu_mass:
        n_elements += 1

    tensions = np.zeros((n_elements, 3))
    tensions[0, 0] = np.cos(beta_n)*np.cos(phi_n)*tension_ground
    tensions[0, 1] = np.cos(beta_n)*np.sin(phi_n)*tension_ground
    tensions[0, 2] = np.sin(beta_n)*tension_ground

    positions = np.zeros((n_elements+1, 3))
    if elastic_elements:
        l_s = (tension_ground/tether_stiffness+1)*l_unstrained
        print((tension_ground/tether_stiffness+1))
    else:
        l_s = l_unstrained
    positions[1, 0] = np.cos(beta_n)*np.cos(phi_n)*l_s
    positions[1, 1] = np.cos(beta_n)*np.sin(phi_n)*l_s
    positions[1, 2] = np.sin(beta_n)*l_s

    velocities = np.zeros((n_elements+1, 3))
    accelerations = np.zeros((n_elements+1, 3))
    non_conservative_forces = np.zeros((n_elements, 3))

    stretched_tether_length = l_s  # Stretched
    for j in range(n_elements):  # Iterate over point masses.
        last_element = j == n_elements - 1
        kcu_element = separate_kcu_mass and j == n_elements - 2

        # Determine kinematics at point mass j.
        vj = np.cross(omega, positions[j+1, :])
        velocities[j+1, :] = vj
        aj = np.cross(omega, vj)
        accelerations[j+1, :] = aj
        delta_p = positions[j+1, :] - positions[j, :]
        ej = delta_p/np.linalg.norm(delta_p)  # Axial direction of tether element

        # Determine flow at point mass j.
        vaj = vj - np.array([vwx, 0, 0])  # Apparent wind velocity
        vajp = np.dot(vaj, ej)*ej  # Parallel to tether element
        # TODO: check whether to use vajn
        vajn = vaj - vajp  # Perpendicular to tether element

        vaj_sq = np.linalg.norm(vaj)*vaj
        # vaj_sq = np.linalg.norm(vajn)*vajn
        tether_drag_basis = rho*l_unstrained*d_t*cd_t*vaj_sq

        # Determine drag at point mass j.
        if not separate_kcu_mass:
            if n_tether_elements == 1:
                dj = -.125*tether_drag_basis
            elif last_element:
                dj = -.25*tether_drag_basis  # TODO: add bridle drag
            else:
                dj = -.5*tether_drag_basis
        else:
            if last_element:
                dj = 0  # TODO: add bridle drag
            elif n_tether_elements == 1:
                dj = -.125*tether_drag_basis
                dj += -.5*rho*vaj_sq*cd_kcu*frontal_area_kcu  # Adding kcu drag
            elif kcu_element:
                dj = -.25*tether_drag_basis
                dj += -.5*rho*vaj_sq*cd_kcu*frontal_area_kcu  # Adding kcu drag
            else:
                dj = -.5*tether_drag_basis

        if not separate_kcu_mass:
            if last_element:
                point_mass = m_s/2 + m_kite + m_kcu
            else:
                point_mass = m_s
        else:
            if last_element:
                point_mass = m_kite
            elif kcu_element:
                point_mass = m_s/2 + m_kcu
            else:
                point_mass = m_s

        # Use force balance to infer tension on next element.
        fgj = np.array([0, 0, -point_mass*g])
        next_tension = point_mass*aj + tensions[j, :] - fgj - dj  # a_kite gave better fit
        if not last_element:
            tensions[j+1, :] = next_tension
            non_conservative_forces[j, :] = dj
        else:
            aerodynamic_force = next_tension
            non_conservative_forces[j, :] = dj + aerodynamic_force

        if ax_plot_forces and not last_element:
            forces = [point_mass*aj, dj, fgj, -tensions[j, :], next_tension]
            labels = ['resultant', 'drag', 'weight', 'last tension', 'next tension']
            clrs = ['m', 'r', 'k', 'g', 'b']
            for f, lbl, clr in zip(forces, labels, clrs):
                # print("{} = {:.2f} N".format(lbl, np.linalg.norm(f)))
                plot_vector(positions[j+1, :], f, ax_plot_forces, color=clr)

        # Derive position of next point mass from former tension
        if kcu_element:
            positions[j+2, :] = positions[j+1, :] + tensions[j+1, :]/np.linalg.norm(tensions[j+1, :]) * l_bridle
        elif not last_element:
            if elastic_elements:
                l_s = (np.linalg.norm(tensions[j+1, :])/tether_stiffness+1)*l_unstrained
            else:
                l_s = l_unstrained
            stretched_tether_length += l_s
            positions[j+2, :] = positions[j+1, :] + tensions[j+1, :]/np.linalg.norm(tensions[j+1, :]) * l_s

    if return_values == 2:
        return positions, velocities, accelerations, tensions, aerodynamic_force, non_conservative_forces
    elif return_values:
        va = vj - np.array([vwx, 0, 0])  # All y-axes are defined perpendicular to apparent wind velocity.

        ez_bridle = tensions[-1, :]/np.linalg.norm(tensions[-1, :])
        ey_bridle = np.cross(ez_bridle, va)/np.linalg.norm(np.cross(ez_bridle, va))
        ex_bridle = np.cross(ey_bridle, ez_bridle)
        dcm_b2e = np.vstack(([ex_bridle], [ey_bridle], [ez_bridle])).T

        ez_tether = tensions[-2, :]/np.linalg.norm(tensions[-2, :])
        ey_tether = np.cross(ez_tether, va)/np.linalg.norm(np.cross(ez_tether, va))
        ex_tether = np.cross(ey_tether, ez_tether)
        dcm_t2e = np.vstack(([ex_tether], [ey_tether], [ez_tether])).T

        ez_f_aero = aerodynamic_force/np.linalg.norm(aerodynamic_force)
        ey_f_aero = np.cross(ez_f_aero, va)/np.linalg.norm(np.cross(ez_f_aero, va))
        ex_f_aero = np.cross(ey_f_aero, ez_f_aero)
        dcm_fa2e = np.vstack(([ex_f_aero], [ey_f_aero], [ez_f_aero])).T

        ez_tau = r_kite/np.linalg.norm(r_kite)
        ey_tau = np.cross(ez_tau, va)/np.linalg.norm(np.cross(ez_tau, va))
        ex_tau = np.cross(ey_tau, ez_tau)
        dcm_tau2e = np.vstack(([ex_tau], [ey_tau], [ez_tau])).T

        return positions, stretched_tether_length, dcm_b2e, dcm_t2e, dcm_fa2e, \
               dcm_tau2e, aerodynamic_force, va
    else:
        return positions[-1, :] - r_kite


def find_tether_length():
    from scipy.optimize import least_squares

    args = (1000, 10, [200, 0, 100], [2, 20, 1], [0, 0, 0])
    opt_res = least_squares(get_tether_end_position, (20 * np.pi / 180., -15 * np.pi / 180., 250), args=args, verbose=2)
    print("Resulting tether length:", opt_res.x[2])
    p = get_tether_end_position(opt_res.x, *args, return_values=True)[0]

    plt.figure(figsize=(8, 6))
    ax3d = plt.axes(projection='3d')

    ax3d.plot(p[:, 0], p[:, 1], p[:, 2], '-s')
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")

    ax3d.set_xlim([0, 250])
    ax3d.set_ylim([-125, 125])
    ax3d.set_zlim([0, 250])

    plt.show()


def find_tether_lengths(flight_data, config, plot=False):
    phi_upwind_direction = -flight_data.loc[flight_data.index[0], 'est_upwind_direction']-np.pi/2.

    from scipy.optimize import least_squares
    dyn = derive_tether_model_kcu_williams(config['n_tether_elements'], False, vwx=vwx)

    if plot:
        # plt.figure(figsize=(3.6, 3.6))
        fig, ax3d = plt.subplots(1, 2, figsize=(7.2, 3.6))
        plt.subplots_adjust(top=1.0, bottom=0.03, left=0.025, right=0.98, hspace=0.2, wspace=0.55)
        ax3d[0] = plt.subplot(121, projection='3d', proj_type='ortho')
        ax3d[0].plot3D(flight_data.rx, flight_data.ry, flight_data.rz, color='k')
        ax3d[0].set_xlim([0, 250])
        ax3d[0].set_ylim([-125, 125])
        ax3d[0].set_zlim([0, 250])
        ax3d[0].set_xlabel(r'$x_{\rm w}$ [m]')
        ax3d[0].set_ylabel(r'$y_{\rm w}$ [m]')
        ax3d[0].set_zlabel(r'$z_{\rm w}$ [m]')
        ax3d[0].set_box_aspect([1, 1, 1])  # As of matplotlib 3.3.0

        ax3d[1].plot(flight_data.rx, flight_data.ry, color='k')
        # ax3d[1].set_xlim([0, 260])
        # ax3d[1].set_ylim([-130, 130])
        ax3d[1].set_xlabel(r'$x_{\rm w}$ [m]')
        ax3d[1].set_ylabel(r'$y_{\rm w}$ [m]')
        ax3d[1].grid()
        ax3d[1].set_aspect('equal')
        add_panel_labels(ax3d, (.05, .35))

    tether_lengths = []
    strained_tether_lengths = []

    n_rows = flight_data.shape[0]
    ypr_bridle = np.empty((n_rows, 3))
    ypr_bridle_vk = np.empty((n_rows, 3))
    ypr_tether = np.empty((n_rows, 3))
    ypr_aero_force = np.empty((n_rows, 3))
    aero_force_body = np.empty((n_rows, 3))
    aero_force_bridle = np.empty((n_rows, 3))
    apparent_flow_direction = np.empty((n_rows, 2))
    kite_front_wrt_projected_velocity = np.empty(n_rows)
    ypr_body2bridle = np.empty((n_rows, 3))

    n_elements = config['n_tether_elements']
    if config['separate_kcu_mass']:
        n_elements += 1

    pos_tau = np.zeros((n_rows, n_elements+1, 3))
    pos_tau_vk = np.zeros((n_rows, n_elements+1, 3))

    mp_counter = 0
    for i, (idx, row) in enumerate(flight_data.iterrows()):
        # Find consistent tether shape
        r_kite = np.array(list(row[['rx', 'ry', 'rz']]))
        v_kite = np.array(list(row[['vx', 'vy', 'vz']]))
        args = (row['ground_tether_force'], config['n_tether_elements'], r_kite,
                list(row[['omx_opt', 'omy_opt', 'omz_opt']]),
                config['separate_kcu_mass'], config['elastic_elements'])
        opt_res = least_squares(get_tether_end_position, list(row[['kite_elevation', 'kite_azimuth', 'kite_distance']]), args=args, verbose=0)
        if not opt_res.success:
            print("Optimization {} failed!".format(i))
        tether_lengths.append(opt_res.x[2])
        pos_e, l_strained, dcm_b2w_i, dcm_t2w_i, dcm_fa2w_i, dcm_tau2w_i, f_aero, v_app = get_tether_end_position(opt_res.x, *args, return_values=True)
        strained_tether_lengths.append(l_strained)

        ez_tau_vk = r_kite/np.linalg.norm(r_kite)
        ey_tau_vk = np.cross(ez_tau_vk, v_kite)/np.linalg.norm(np.cross(ez_tau_vk, v_kite))
        ex_tau_vk = np.cross(ey_tau_vk, ez_tau_vk)
        dcm_e2tau_vk_i = np.vstack(([ex_tau_vk], [ey_tau_vk], [ez_tau_vk]))

        for j in range(pos_e.shape[0]):
            pos_tau[i, j, :] = dcm_tau2w_i.T.dot(pos_e[j, :])
            pos_tau_vk[i, j, :] = dcm_e2tau_vk_i.dot(pos_e[j, :])

        dcm_tau2b = dcm_b2w_i.T.dot(dcm_tau2w_i)
        # Note that dcm_b2e_i and dcm_tau2w_i both use the apparent wind velocity to define the positive x-axis, as such
        # the yaw should be roughly zero and it should not matter where to put the 3-rotation when unravelling.
        ypr_bridle[i, :] = unravel_euler_angles(dcm_tau2b, '321')

        dcm_tau_vk2b = dcm_b2w_i.T.dot(dcm_e2tau_vk_i.T)
        # Note that if we use 3-2-1 sequence here, we'll find the same roll as for the upper as the tau ref frame is
        # just rotated along the z-axis wrt tau_vk
        ypr_bridle_vk[i, :] = unravel_euler_angles(dcm_tau_vk2b, '213')

        dcm_tau2t = dcm_t2w_i.T.dot(dcm_tau2w_i)
        ypr_tether[i, :] = unravel_euler_angles(dcm_tau2t, '321')

        dcm_tau2fa = dcm_fa2w_i.T.dot(dcm_tau2w_i)
        ypr_aero_force[i, :] = unravel_euler_angles(dcm_tau2fa, '321')

        dcm_wind2body = rotation_matrix_earth2body(row['roll'], row['pitch'], row['yaw']-phi_upwind_direction)
        aero_force_body[i, :] = dcm_wind2body.dot(f_aero)
        aero_force_bridle[i, :] = dcm_b2w_i.T.dot(f_aero)
        
        v_app_body = dcm_wind2body.dot(v_app)
        apparent_flow_direction[i, 0] = -np.arctan2(v_app_body[2], v_app_body[0])
        apparent_flow_direction[i, 1] = np.arctan2(v_app_body[1], v_app_body[0])

        kite_front_wrt_projected_velocity[i] = calc_kite_front_wrt_projected_velocity(r_kite, v_kite, dcm_wind2body)

        dcm_body2b = dcm_b2w_i.T.dot(dcm_wind2body.T)
        ypr_body2bridle[i, :] = unravel_euler_angles(dcm_body2b, '321')

        if plot and i in mark_points:
            clr = 'C{}'.format(mp_counter)
            ax3d[0].plot3D(pos_e[:, 0], pos_e[:, 1], pos_e[:, 2], color=clr)  #, linewidth=.5, color='grey')
            ax3d[1].plot(pos_e[:, 0], pos_e[:, 1], color=clr, label=mp_counter + 1)
            # Cross-heading is preferred opposed to cross-course as it shows a helix shape of the tether at the outside
            # of the turns, which one could interpret as caused by the centripetal acceleration, however it can be
            # to the drag.
            mp_counter += 1

        verify = False
        if verify:
            pos_e, v, a, t, fa, fnc = get_tether_end_position(opt_res.x, *args, return_values=2)
            x = np.vstack((pos_e[1:, :].reshape((-1, 1)), v[1:, :].reshape((-1, 1)), [[opt_res.x[2]], [0]]))
            u = [0, *a[-1, :]]
            dp = pos_e[1:, :] - pos_e[:-1, :]
            nu = t[:, 0]/dp[:, 0]
            b = np.hstack((a[1:, :].reshape(-1), nu))

            # Checking residual when filling in system of eqs.
            c, d = dyn['f_mat'](x, u)
            eps = np.array(c@b - d)
            assert np.amax(np.abs(eps)) < 1e-6
    if plot:
        ax3d[1].legend(title='Instance label', bbox_to_anchor=(.0, 1.05, 1., .5), loc="lower left", mode="expand",
                       borderaxespad=0, ncol=5)

    if config['separate_kcu_mass']:
        tether_lengths = np.array(tether_lengths) + l_bridle
        strained_tether_lengths = np.array(strained_tether_lengths) + l_bridle

    return tether_lengths, strained_tether_lengths, ypr_bridle, ypr_bridle_vk, ypr_tether, ypr_aero_force,\
           aero_force_body, aero_force_bridle, apparent_flow_direction, kite_front_wrt_projected_velocity, \
           ypr_body2bridle, pos_tau


def find_tether_forces(flight_data, tether_lengths, config):
    assert not config['elastic_elements']
    from scipy.optimize import least_squares
    tether_forces = []
    ypr_tether_end = np.empty((flight_data.shape[0], 3))
    ypr_tether_second_last = np.empty((flight_data.shape[0], 3))
    ypr_aero_force = np.empty((flight_data.shape[0], 3))
    for i, (idx, row) in enumerate(flight_data.iterrows()):
        args = (tether_lengths[i], config['n_tether_elements'], list(row[['rx', 'ry', 'rz']]),
                list(row[['vx', 'vy', 'vz']]), list(row[['ax', 'ay', 'az']]),
                config['separate_kcu_mass'], config['elastic_elements'])
        opt_res = least_squares(get_tether_end_position, list(row[['kite_elevation', 'kite_azimuth', 'kite_distance']]), args=args,
                                kwargs={'find_force': True}, verbose=0)
        if not opt_res.success:
            print("Optimization failed!")
        tether_forces.append(opt_res.x[2])
        p, l_strained, dcm_t2e_i, dcm_tsl2e_i, dcm_fa2e_i, dcm_tau2e_i = get_tether_end_position(opt_res.x, *args, return_values=True,
                                                                                                 find_force=True)
        dcm_tau2t = dcm_t2e_i.T.dot(dcm_tau2e_i)
        ypr_tether_end[i, :] = unravel_euler_angles(dcm_tau2t, '321')

        dcm_tau2tsl = dcm_tsl2e_i.T.dot(dcm_tau2e_i)
        ypr_tether_second_last[i, :] = unravel_euler_angles(dcm_tau2tsl, '321')

        dcm_tau2fa = dcm_fa2e_i.T.dot(dcm_tau2e_i)
        ypr_aero_force[i, :] = unravel_euler_angles(dcm_tau2fa, '321')
    return tether_forces, ypr_tether_end, ypr_tether_second_last, ypr_aero_force


def plot_tether_element_pitch(flight_data, ypr_bridle, ax):
    # pitch_powered = get_pitch_nose_down_angle_v3(.78)
    # flight_data['depower_pitch'] = get_pitch_nose_down_angle_v3(1-flight_data['kite_actual_depower']/100.) - pitch_powered

    ax.plot(flight_data.time, ypr_bridle[:, 1]*180./np.pi, label='T-I N=30')
    ax.grid()
    ax.plot(flight_data.time, flight_data.pitch0_tau * 180. / np.pi, label='Sensor 0')  #-flight_data['depower_pitch'])
    ax.plot(flight_data.time, flight_data.pitch1_tau * 180. / np.pi, label='Sensor 1')
    ax.set_ylim([0, None])
    ax.set_xlim([flight_data.iloc[0]['time'], flight_data.iloc[-1]['time']])
    plot_flight_sections(ax, flight_data)


def plot_tether_element_attitudes(flight_data, ypr_aero_force, ypr_bridle, ypr_tether, separate_kcu_mass):
    plot_yaw = True
    if plot_yaw:
        n_rows = 3
    else:
        n_rows = 2
    fig, ax_ypr = plt.subplots(n_rows, 1, sharex=True)
    plt.suptitle("3-2-1 Euler angles between tangential\nand local ref. frames")
    # ax_ypr[0].plot(flight_data.time, ypr_aero_force[:, 1]*180./np.pi, label='Aero force')
    if separate_kcu_mass:
        ax_ypr[0].plot(flight_data.time, ypr_bridle[:, 1]*180./np.pi, label='Bridle')
        # ax_ypr[0].plot(flight_data.time, ypr_tether[:, 1]*180./np.pi, label='Tether')
    else:
        ax_ypr[0].plot(flight_data.time, ypr_bridle[:, 1]*180./np.pi, label='Tether')
    ax_ypr[0].set_ylabel("Pitch [deg]")

    ax_ypr[1].plot(flight_data.time, ypr_aero_force[:, 2]*180./np.pi, label='Aero force')
    if separate_kcu_mass:
        clr = ax_ypr[1].plot(flight_data.time, ypr_bridle[:, 2]*180./np.pi, label='Bridle')[0].get_color()
        # ax_ypr[1].plot(flight_data.time, ypr_bridle_vk[:, 2]*180./np.pi, '--', label='sim bridle vk', color=clr)
        ax_ypr[1].plot(flight_data.time, ypr_tether[:, 2]*180./np.pi, label='Tether')
    else:
        ax_ypr[1].plot(flight_data.time, ypr_bridle[:, 2]*180./np.pi, label='Tether')
    ax_ypr[1].set_ylabel("Roll [deg]")

    if plot_yaw:
        ax_ypr[2].plot(flight_data.time, ypr_bridle[:, 0]*180./np.pi, label='Bridle')
        ax_ypr[2].set_ylabel("Yaw [deg]")

    ax_ypr[-1].set_xlabel("Time [s]")

    for a in ax_ypr: a.grid()
    clr0 = ax_ypr[0].plot(flight_data.time, flight_data.pitch0_tau * 180. / np.pi, label='Sensor 0')[0].get_color()
    clr1 = ax_ypr[0].plot(flight_data.time, flight_data.pitch1_tau * 180. / np.pi, label='Sensor 1')[0].get_color()
    clr2 = ax_ypr[0].plot(flight_data.time, flight_data.pitch_tau * 180. / np.pi, label='Sensor avg')[0].get_color()
    # ax_ypr[1].plot(flight_data.time, flight_data.roll0_tau * 180. / np.pi, label='Sensor 0')
    # ax_ypr[1].plot(flight_data.time, flight_data.roll1_tau * 180. / np.pi, label='Sensor 1')
    ax_ypr[1].plot(flight_data.time, flight_data.roll_tau * 180. / np.pi, label='Sensor avg')
    ax_ypr[1].legend()
    # if plot_yaw:
    #     ax_ypr[2].plot(flight_data.time, flight_data.yaw0_tau * 180. / np.pi, label='Sensor 0', color=clr0)
    #     ax_ypr[2].plot(flight_data.time, flight_data.yaw1_tau * 180. / np.pi, label='Sensor 1', color=clr1)
    #     ax_ypr[2].plot(flight_data.time, (np.pi - flight_data.kite_heading) * 180. / np.pi, label='Heading')
    #     ax_ypr[2].plot(flight_data.time, (np.pi - flight_data.kite_course) * 180. / np.pi, label='Course')

    ax_ypr[0].set_ylim([0, None])

    ax_ypr[2].plot(flight_data.time, flight_data.kite_actual_steering, '--')

    for a in ax_ypr: plot_flight_sections(a, flight_data)


def plot_aero_force_components(flight_data, aero_force_body, aero_force_bridle, flow_angles, ypr_body2bridle,
                               kite_front_wrt_projected_velocity):
    fig, ax = plt.subplots(3, 2)
    ax[0, 0].plot(flight_data.time, aero_force_body[:, 0], label='body')
    ax[0, 0].plot(flight_data.time, aero_force_bridle[:, 0], label='bridle')
    ax[0, 0].legend()
    ax[0, 0].set_ylabel("$F_{aero,x}$ [N]")
    ax[1, 0].plot(flight_data.time, aero_force_body[:, 1])
    ax[1, 0].plot(flight_data.time, aero_force_bridle[:, 1])
    # ax[1, 0].plot(flight_data.time, flight_data.kite_set_steering*1e1, label='set steering')
    ax[1, 0].plot(flight_data.time, flight_data.kite_actual_steering*1e1, label='actual steering')
    ax[1, 0].legend()
    ax[1, 0].set_ylabel("$F_{aero,y}$ [N]")
    ax[2, 0].plot(flight_data.time, aero_force_body[:, 2])
    ax[2, 0].plot(flight_data.time, aero_force_bridle[:, 2])
    ax[2, 0].set_ylabel("$F_{aero,z}$ [N]")
    ax[2, 0].set_xlabel("Time [s]")

    ax[0, 1].plot(flight_data.time, flow_angles[:, 0]*180/np.pi)
    ax[0, 1].plot(flight_data.time, ypr_body2bridle[:, 1]*180/np.pi)
    ax[0, 1].set_ylabel(r"$\alpha$ [deg]")
    ax[1, 1].plot(flight_data.time, flow_angles[:, 1]*180/np.pi)
    ax[1, 1].plot(flight_data.time, ypr_body2bridle[:, 0]*180/np.pi)
    ax[1, 1].set_ylabel(r"$\beta$ [deg]")
    ax[2, 1].plot(flight_data.time, -kite_front_wrt_projected_velocity*180/np.pi)
    ax[2, 1].set_ylabel(r"$\beta to v_{kite}$ [deg]")


def plot_tether_states(flight_data, tether_lengths, strained_tether_lengths=None,
                       fitted_tether_length_and_speed=None, fitted_tether_acceleration=None):
    # Tether lengths include bridle length
    if strained_tether_lengths is None:
        strained_and_unstrained_same = True
        strained_tether_lengths = tether_lengths
    else:
        strained_and_unstrained_same = False

    fig, ax2 = plt.subplots(1, 2, sharex=True, figsize=[6.4, 2.4])
    plt.subplots_adjust(top=0.97, bottom=0.2, left=0.14, right=0.99, wspace=0.4)
    ax2[0].plot(flight_data.time[:-1], np.diff(strained_tether_lengths)/.1, linewidth=.5, label='T-I')
    ax2[0].plot(flight_data.time, flight_data.ground_tether_reelout_speed, label='Measured')
    ax2[0].set_ylabel(r'Tether speed [m s$^{-1}$]')
    ax2[0].set_ylim([0, 2.5])
    ax2[1].set_ylabel(r'Tether acceleration [m s$^{-2}$]')
    ax2[0].set_xlabel('Time [s]')
    ax2[1].set_xlabel('Time [s]')
    for a in ax2: a.grid()
    for a in ax2: a.set_xlim([flight_data.iloc[0]['time'], flight_data.iloc[-1]['time']])

    fig, ax = plt.subplots(3, 2, sharex=True)
    ax[0, 0].plot(flight_data.time, np.array(strained_tether_lengths)-flight_data.kite_distance, label='sag')
    # coef = np.polyfit(flight_data.time, flight_data.kite_distance, 1)
    # poly1d_fn = np.poly1d(coef)
    # delta_linear_fun = flight_data.kite_distance - poly1d_fn(flight_data.time)
    # ax[0].plot(flight_data.time, delta_linear_fun, label='delta r wrt linear')
    ax[0, 0].legend()
    # ax[0].plot(flight_data.time, np.array(tether_lengths)-flight_data.kite_distance, ':')
    ax[0, 0].fill_between(flight_data.time, 0, 1, where=flight_data['flag_turn'], facecolor='lightsteelblue', alpha=0.5)
    ax[0, 0].set_ylabel('Sag length [m]')

    ax[1, 0].plot(flight_data.time, flight_data.ground_tether_force*1e-3)
    ax[1, 0].fill_between(flight_data.time, 0, 5, where=flight_data['flag_turn'], facecolor='lightsteelblue', alpha=0.5)
    ax[1, 0].set_ylabel('Tether force [kN]')

    ax[2, 0].plot(flight_data.time, strained_tether_lengths, label='strained')
    if not strained_and_unstrained_same:
        ax[2, 0].plot(flight_data.time, tether_lengths, ':', label='unstrained')
    ax[2, 0].plot(flight_data.time, flight_data.kite_distance, '-.', label='radial position')
    # bias = np.mean(tether_lengths - flight_data.ground_tether_length)
    # print("Tether length bias", bias)
    # ax[1].plot(flight_data.time, flight_data.ground_tether_length + bias, '-.', label='mea')
    ax[2, 0].legend()
    ax[2, 0].set_ylabel('Tether length [m]')

    ax[0, 1].plot(flight_data.time[:-1], np.diff(strained_tether_lengths)/.1, linewidth=.5)
    ax[0, 1].plot(flight_data.time, flight_data.ground_tether_reelout_speed, ':')
    ax[0, 1].set_ylabel('Tether speed [m/s]')
    ax[0, 1].set_ylim([0, 2.5])

    ax[1, 1].set_ylabel('Tether acceleration [m/s**2]')

    ax[2, 1].plot(flight_data.time[:-1], -np.diff(np.array(strained_tether_lengths)-flight_data.kite_distance)/.1, label='sag speed')
    stretch = flight_data.ground_tether_force/tether_stiffness*flight_data.kite_distance
    ax[2, 1].plot(flight_data.time[:-1], np.diff(stretch)/.1, label='stretch speed')

    if fitted_tether_length_and_speed is not None:
        print("Start tether speed", fitted_tether_length_and_speed[0, 1])
        ax[0, 0].plot(flight_data.time, fitted_tether_length_and_speed[:, 0]-flight_data.kite_distance, '--')
        ax[2, 0].plot(flight_data.time, fitted_tether_length_and_speed[:, 0], '--')
        ax[0, 1].plot(flight_data.time, fitted_tether_length_and_speed[:, 1], '--')
        ax2[0].plot(flight_data.time, fitted_tether_length_and_speed[:, 1], label='Fit')
        ax[1, 1].plot(flight_data.time[:-1], fitted_tether_acceleration)
        ax2[1].plot(flight_data.time[:-1], fitted_tether_acceleration)
    ax2[0].legend(ncol=2)
    add_panel_labels(ax2, offset_x=.36)


def plot_offaxial_tether_displacement(pos_tau, ax=None, ls='-', plot_rows=[0, 1], plot_instances=mark_points[:5]):
    if ax is None:
        set_legend = True

        fig, ax = plt.subplots(2, 2, sharey=True, figsize=(6.4, 5.2))
        ax[0, 1].invert_xaxis()
        ax[1, 1].invert_xaxis()
        wspace = .200
        plt.subplots_adjust(top=0.885, bottom=0.105, left=0.15, right=0.99, hspace=0.07, wspace=wspace)
        ax[0, 0].set_ylabel("Radial position [m]")
        ax[1, 0].set_ylabel("Radial position [m]")
        ax[1, 0].set_xlabel("Up-heading position [m]")
        ax[1, 1].set_xlabel("Cross-heading position [m]")
        ax[1, 0].set_xlim([-8.5, 1.5])
        ax[1, 1].set_xlim([5, -5])
        ax[1, 0].get_shared_x_axes().join(*ax[:, 0])
        ax[1, 1].get_shared_x_axes().join(*ax[:, 1])
        for a in ax.reshape(-1): a.grid()
        ax[0, 0].tick_params(labelbottom=False)
        ax[0, 1].tick_params(labelbottom=False)
    else:
        set_legend = False

    for ir in plot_rows:
        for counter, i in enumerate(plot_instances):
            # if mp_counter in [0, 2, 3]:
            clr = 'C{}'.format(counter)
            ax[ir, 0].plot(pos_tau[i, :, 0], pos_tau[i, :, 2], ls, color=clr, label='{}'.format(counter+1))
            ax[ir, 1].plot(pos_tau[i, :, 1], pos_tau[i, :, 2], ls, color=clr)

    if set_legend:
        ax[0, 0].legend(title='Instance label', bbox_to_anchor=(.2, 1.05, 1.6+wspace, .5), loc="lower left", mode="expand",
                     borderaxespad=0, ncol=5)
    return ax


def find_and_plot_tether_lengths(n_tether_elements=30, export_tether_lengths=False, i_cycle=None, ax=None, config=None, input_file_suffix=''):
    if config is None:
        config = {
            'n_tether_elements': n_tether_elements,
            'separate_kcu_mass': True,
            'elastic_elements': False,
            'make_kinematics_consistent': True,
        }

    flight_data = read_and_transform_flight_data(config['make_kinematics_consistent'], i_cycle, input_file_suffix)  # Read flight data.
    if i_cycle is not None:
        flight_data.loc[flight_data['phase'] >= 3, 'azimuth_turn_center'] = np.nan
        flight_data.loc[flight_data['phase'] >= 3, 'elevation_turn_center'] = np.nan
        flight_data.loc[flight_data['phase'] >= 3, 'flag_turn'] = False
    determine_rigid_body_rotation(flight_data)

    tether_lengths, strained_tether_lengths, ypr_bridle, ypr_bridle_vk, ypr_tether, ypr_aero_force, \
    aero_force_body, aero_force_bridle, flow_angles, kite_front_wrt_projected_velocity, ypr_body2bridle, pos_tau = \
        find_tether_lengths(flight_data, config, plot=False)

    if ax is None:
        plot_offaxial_tether_displacement(pos_tau)
        plot_aero_force_components(flight_data, aero_force_body, aero_force_bridle, flow_angles, ypr_body2bridle,
                                   kite_front_wrt_projected_velocity)
        plot_tether_element_attitudes(flight_data, ypr_aero_force, ypr_bridle, ypr_tether, config['separate_kcu_mass'])
    else:
        assert config['separate_kcu_mass'], "Analysis only used with bridle element"
        np.save('ypr_bridle_cycle{}.npy'.format(i_cycle), ypr_bridle)
        plot_tether_element_pitch(flight_data, ypr_bridle, ax)

        # fitted_tether_length_and_speed, fitted_tether_acceleration = \
        #     match_tether_length_and_speed(tether_lengths, flight_data.ground_tether_reelout_speed,
        #                                   flight_data.kite_distance)
        # plot_tether_states(flight_data, strained_tether_lengths, tether_lengths,
        #                    fitted_tether_length_and_speed, fitted_tether_acceleration)

    if export_tether_lengths:
        np.save('tether_lengths_{}.npy'.format(input_file_suffix), tether_lengths)
        # if config['separate_kcu_mass']:
        #     np.save('ypr_bridle_rigid_body_rotation.npy', ypr_bridle)

    if config['separate_kcu_mass'] and not config['elastic_elements']:
        res = {
            'strained_tether_lengths': strained_tether_lengths,
            'pitch_bridle': ypr_bridle[:, 1],
            'roll_bridle': ypr_bridle[:, 2],
            'offaxial_tether_shape': pos_tau,
        }
        with open("time_invariant_results{}.pickle".format(config['n_tether_elements']), 'wb') as f:
            pickle.dump(res, f)


def match_tether_length_and_speed(tether_lengths, tether_speeds, radius):
    import casadi as ca

    n_intervals = len(tether_lengths)-1
    tf = .1

    # ODE for kinematic model
    l = ca.SX.sym('l')
    dl = ca.SX.sym('dl')
    ddl = ca.SX.sym('ddl')

    x = ca.vertcat(l, dl)
    ode = ca.vertcat(dl, ddl)

    # Create an integrator
    dae = {'x': x, 'ode': ode, 'p': ddl}

    intg = ca.integrator('intg', 'idas', dae, {'tf': tf})

    opti = ca.casadi.Opti()

    # Decision variables for states
    states = opti.variable(n_intervals+1, 2)
    # Decision variables for control vector
    controls = opti.variable(n_intervals)

    # Gap-closing shooting constraints
    for k in range(n_intervals):
        res = intg(x0=states[k, :], p=controls[k])
        opti.subject_to(states[k+1, :].T == res["xf"])

    obj = ca.sumsqr(states[:, 0] - np.array(tether_lengths))

    # # Lower objective finds
    # obj += ca.sumsqr(.1*(states[:, 1] - np.array(tether_speeds)))
    # obj = obj + ca.sumsqr(states[n_intervals, 0] - tether_lengths[n_intervals])

    # control_steps = controls[1:, :]-controls[:-1, :]
    # weight_control_steps = 1  # sumsqr of control steps resulting with given weights: {1/15: 3606, 1/30: 5526, 1/100: 12315, 0: 21295}
    # obj += ca.sumsqr(weight_control_steps*control_steps)

    opti.minimize(obj)
    opti.subject_to(opti.bounded(-.3, controls[1:]-controls[:-1], .3))
    opti.subject_to(states[:, 0] - radius > .01)

    # solve optimization problem
    opti.solver('ipopt')

    sol = opti.solve()
    print(sol.value(obj))
    # print(sol.value(ca.sumsqr(.1*(states[:, 1] - np.array(tether_speeds)))))
    # print(sol.value(ca.sumsqr(weight_control_steps*control_steps)))

    return sol.value(states), sol.value(controls)


def combine_results_of_different_analyses():
    flight_data = read_and_transform_flight_data()

    fig, ax_ypr = plt.subplots(2, 1, sharex=True, figsize=[6.4, 4])
    plt.subplots_adjust(top=0.85, bottom=0.135, left=0.16, right=0.99,)
    # plt.suptitle("3-2-1 Euler angles between tangential\nand bridle ref. frames")
    ax_ypr[0].set_ylabel("Pitch [$^\circ$]")
    ax_ypr[1].set_ylabel("Roll [$^\circ$]")
    ax_ypr[1].set_xlabel("Time [s]")

    fig, ax_tether_length = plt.subplots(3, 1, sharex=True)

    n_tether_elements = [30, 1]
    linestyles = ['-', '--']
    linewidths = [2.5, 1.5]
    ax_tether_shape = None
    for n_te, ls, lw in zip(n_tether_elements, linestyles, linewidths):
        with open("time_invariant_results{}.pickle".format(n_te), 'rb') as f:
            res = pickle.load(f)
        if n_te == 1:
            plot_rows = [0]
        else:
            plot_rows = [0, 1]
        ax_tether_shape = plot_offaxial_tether_displacement(res['offaxial_tether_shape'], ax_tether_shape, ls=ls,
                                                            plot_rows=plot_rows)

        ax_ypr[0].plot(flight_data.time, res['pitch_bridle']*180./np.pi, ls=ls, linewidth=lw, label=r'T-I N='+str(n_te))
        ax_ypr[1].plot(flight_data.time, res['roll_bridle']*180./np.pi, ls=ls, linewidth=lw, label=r'T-I N='+str(n_te))

        ax_tether_length[0].plot(flight_data.time, res['strained_tether_lengths'], label=r'T-I N='+str(n_te))
        ax_tether_length[1].plot(flight_data.time, res['strained_tether_lengths']-flight_data.kite_distance)
    ax_tether_length[0].plot(flight_data.time, flight_data.kite_distance)
    ax_tether_length[0].legend()
    add_panel_labels(ax_tether_length)

    with open("dynamic_results30.pickle", 'rb') as f:
        res = pickle.load(f)
    ax_tether_shape = plot_offaxial_tether_displacement(res['offaxial_tether_shape'], ax_tether_shape, ls='--', plot_rows=[1])
    ax_ypr[0].plot(flight_data.time, res['pitch_bridle']*180./np.pi, label=r'Dyn N=30')
    ax_ypr[1].plot(flight_data.time, res['roll_bridle']*180./np.pi, label=r'Dyn N=30')

    ax_ypr[0].plot(flight_data.time, flight_data.pitch0_tau * 180. / np.pi, '-.', label='Sensor 0')
    ax_ypr[0].plot(flight_data.time, flight_data.pitch1_tau * 180. / np.pi, '-.', label='Sensor 1')
    ax_ypr[1].plot(flight_data.time, flight_data.roll0_tau * 180. / np.pi, '-.', label='Sensor 0')
    ax_ypr[1].plot(flight_data.time, flight_data.roll1_tau * 180. / np.pi, '-.', label='Sensor 1')

    ax_ypr[0].legend(bbox_to_anchor=(.15, 1.07, .7, .5), loc="lower left", mode="expand",
                   borderaxespad=0, ncol=3)
    ax_ypr[0].set_ylim([0, None])
    for a in ax_ypr: a.grid()
    for a in ax_ypr: plot_flight_sections(a, flight_data)
    add_panel_labels(ax_ypr, .18)
    ax_ypr[0].set_xlim([flight_data.iloc[0]['time'], flight_data.iloc[-1]['time']])

    add_panel_labels(ax_tether_shape, (.38, .15))


def sensitivity_study():
    instance_ref = mark_points[2]
    i_cycle = 65
    config = {
        'vwx': vwx,
        'separate_kcu_mass': True,
        'elastic_elements': False,
    }

    flight_data = read_and_transform_flight_data(True, i_cycle)  # Read flight data.
    determine_rigid_body_rotation(flight_data)
    flight_data = flight_data[instance_ref:instance_ref+1]

    ax = None
    ypr_bridle_sensitivity = np.empty((0, 3))
    n_elem_list = [15, 20, 30]
    for n_tether_elements in n_elem_list:
        config['n_tether_elements'] = n_tether_elements

        tether_lengths, strained_tether_lengths, ypr_bridle, ypr_bridle_vk, ypr_tether, ypr_aero_force, \
        aero_force_body, aero_force_bridle, flow_angles, kite_front_wrt_projected_velocity, ypr_body2bridle, pos_tau = \
            find_tether_lengths(flight_data, config, plot=False)
        ypr_bridle_sensitivity = np.vstack((ypr_bridle_sensitivity, ypr_bridle))

        ax = plot_offaxial_tether_displacement(pos_tau, ax, plot_rows=[0], plot_instances=[0])

    fig, ax_ypr = plt.subplots(2, 1, sharex=True)
    plt.suptitle("3-2-1 Euler angles between tangential\nand local ref. frames")
    ax_ypr[0].plot(n_elem_list, ypr_bridle_sensitivity[:, 1]*180./np.pi, 's-', label='Bridle')
    ax_ypr[0].set_ylabel("Pitch [deg]")
    ax_ypr[1].plot(n_elem_list, ypr_bridle_sensitivity[:, 2]*180./np.pi, 's-', label='Bridle')
    ax_ypr[1].set_ylabel("Roll [deg]")
    ax_ypr[1].set_xlabel("Number of tether elements [-]")


def plot_pitch_multi_cycles():
    fig, ax = plt.subplots(5, 2, sharey=True, figsize=[6.4, 6.4])
    left, right, wspace = 0.105, 0.985, 0.085
    plt.subplots_adjust(top=0.945, bottom=0.085, left=left, right=right, hspace=0.36, wspace=0.085)
    fig.supxlabel('Time [s]', x=(left+right)/2)
    fig.supylabel("Pitch [$^\circ$]")
    for ic, a in zip(list(range(65, 66)), ax.reshape(-1)):
        find_and_plot_tether_lengths(30, i_cycle=ic, ax=a)
        a.text(.05, .75, "Cycle {}".format(ic), transform=a.transAxes)
        break
    ax[0, 0].legend(bbox_to_anchor=(.4, 1.07, 1.2+wspace, .5), loc="lower left", mode="expand",
                       borderaxespad=0, ncol=3)
    plt.show()


def sensitivity_kcu():
    i_cycle = 65
    flight_data = read_and_transform_flight_data(False, i_cycle)  # Read flight data.

    fig, ax_ypr = plt.subplots(3, 1, sharex=True, figsize=[6.4, 4])
    plt.subplots_adjust(top=0.85, bottom=0.135, left=0.16, right=0.99,)
    ax_ypr[0].set_ylabel("Pitch [$^\circ$]")
    # ax_ypr[1].set_ylabel("Roll [$^\circ$]")
    ax_ypr[-1].set_xlabel("Time [s]")

    ypr_l = np.load('ypr_bridle_cycle65_large_KCU.npy')
    ypr_s = np.load('ypr_bridle_cycle65_small_KCU.npy')
    ypr_light = np.load('ypr_bridle_cycle65_light_KCU.npy')
    ypr_op = np.load('ypr_bridle_cycle65_original_path.npy')

    ax_ypr[0].plot(flight_data.time, ypr_l[:, 1]*180./np.pi)
    ax_ypr[0].plot(flight_data.time, ypr_s[:, 1]*180./np.pi)
    ax_ypr[0].plot(flight_data.time, flight_data.pitch_tau*180./np.pi)
    # ax_ypr[0].plot(flight_data.time, ypr_light[:, 1]*180./np.pi)
    # ax_ypr[0].plot(flight_data.time, ypr_op[:, 1]*180./np.pi)
    pitch_bias = np.mean(ypr_l[:, 1]-ypr_s[:, 1])
    ax_ypr[0].plot(flight_data.time, (ypr_s[:, 1]+pitch_bias)*180./np.pi, '-.')

    ax_ypr[1].plot(flight_data.time, flight_data.pitch0_tau*180./np.pi)
    ax_ypr[1].plot(flight_data.time, flight_data.pitch1_tau*180./np.pi)
    ax_ypr[1].plot(flight_data.time, flight_data.pitch_tau*180./np.pi)

    ax_ypr[2].plot(flight_data.time, ypr_l[:, 2]*180./np.pi, label="large")
    ax_ypr[2].plot(flight_data.time, ypr_s[:, 2]*180./np.pi, label="small")
    ax_ypr[2].plot(flight_data.time, flight_data.roll_tau*180./np.pi, label="mea")
    ax_ypr[2].plot(flight_data.time, ypr_light[:, 2]*180./np.pi, label="light")
    ax_ypr[2].plot(flight_data.time, ypr_op[:, 2]*180./np.pi, label="original path")
    ax_ypr[2].legend()


def fit_reelout_acceleration_to_lengths(i_cycle, input_file_suffix):
    flight_data = read_and_transform_flight_data(True, i_cycle, input_file_suffix)
    tether_lengths = np.load('tether_lengths_{}.npy'.format(input_file_suffix))

    fitted_tether_length_and_speed, fitted_tether_acceleration = \
        match_tether_length_and_speed(tether_lengths, flight_data.ground_tether_reelout_speed,
                                      flight_data.kite_distance)

    plot_tether_states(flight_data, tether_lengths, fitted_tether_length_and_speed=fitted_tether_length_and_speed,
                       fitted_tether_acceleration=fitted_tether_acceleration)

    fitted_tether_length_and_speed[:, 0] = fitted_tether_length_and_speed[:, 0]-l_bridle
    opt_controls = np.hstack((fitted_tether_acceleration, [np.nan])).reshape((-1, 1))
    save_array = np.hstack((fitted_tether_length_and_speed, opt_controls))
    np.save('tether_states_{}.npy'.format(input_file_suffix), save_array)


if __name__ == "__main__":
    # plot_pitch_multi_cycles()
    # find_and_plot_tether_lengths(1)
    # config = {
    #     'n_tether_elements': 30,
    #     'separate_kcu_mass': True,
    #     'elastic_elements': True,
    #     'make_kinematics_consistent': False,
    # }
    # find_and_plot_tether_lengths(30, i_cycle=65, export_tether_lengths=True, input_file_suffix='rugid')  #, config=config)
    fit_reelout_acceleration_to_lengths(65, 'rugid')
    # find_and_plot_tether_lengths(100)
    # combine_results_of_different_analyses()
    # sensitivity_study()
    # sensitivity_kcu()
    plt.show()
