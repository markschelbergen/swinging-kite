import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from utils import unravel_euler_angles, plot_flight_sections, rotation_matrix_earth2body
from system_properties import *
from tether_model_and_verification import derive_tether_model_kcu_williams
from attitude_check import calc_kite_kite_front_wrt_projected_velocity
from turning_center import mark_points


def plot_vector(p0, v, ax, scale_vector=.03, color='g', label=None):
    p1 = p0 + v * scale_vector
    vector = np.vstack(([p0], [p1])).T
    ax.plot3D(vector[0], vector[1], vector[2], color=color, label=label)


def shoot(x, set_parameter, n_tether_elements, r_kite, omega, vwx, separate_kcu_mass=False, elastic_elements=True, ax_plot_forces=False, return_values=False, find_force=False):
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
    else:
        l_s = l_unstrained
    positions[1, 0] = np.cos(beta_n)*np.cos(phi_n)*l_s
    positions[1, 1] = np.cos(beta_n)*np.sin(phi_n)*l_s
    positions[1, 2] = np.sin(beta_n)*l_s

    velocities = np.zeros((n_elements+1, 3))
    accelerations = np.zeros((n_elements+1, 3))
    non_conservative_forces = np.zeros((n_elements, 3))

    stretched_tether_length = l_s  # Stretched
    for j in range(n_elements):
        last_element = j == n_elements - 1
        kcu_element = separate_kcu_mass and j == n_elements - 2

        vj = np.cross(omega, positions[j+1, :])
        velocities[j+1, :] = vj
        aj = np.cross(omega, vj)
        accelerations[j+1, :] = aj
        delta_p = positions[j+1, :] - positions[j, :]
        ej = delta_p/np.linalg.norm(delta_p)  # Axial direction of tether element

        vaj = vj - np.array([vwx, 0, 0])  # Apparent wind velocity
        vajp = np.dot(vaj, ej)*ej  # Parallel to tether element
        # TODO: check whether to use vajn
        vajn = vaj - vajp  # Perpendicular to tether element

        if separate_kcu_mass and last_element:
            dj = 0  # TODO: add bridle drag
        else:
            # dj = -.5*rho*l_unstrained*d_t*cd_t*np.linalg.norm(vajn)*vajn  # Zanon does not take perpendicular component.
            dj = -.5*rho*l_unstrained*d_t*cd_t*np.linalg.norm(vaj)*vaj

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
                # dj += -.5*rho*1.6*1.*np.linalg.norm(vajn)*vajn  # Adding kcu drag
                dj += -.5*rho*1.6*1.*np.linalg.norm(vaj)*vaj  # Adding kcu drag
            else:
                point_mass = m_s

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

    args = (1000, 10, [200, 0, 100], [2, 20, 1], [0, 0, 0], 10)
    opt_res = least_squares(shoot, (20*np.pi/180., -15*np.pi/180., 250), args=args, verbose=2)
    print("Resulting tether length:", opt_res.x[2])
    p = shoot(opt_res.x, *args, return_values=True)[0]

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


def find_tether_lengths(flight_data, shoot_args, ax):
    dyn = derive_tether_model_kcu_williams(shoot_args['n_tether_elements'], False, vwx=shoot_args['vwx'])

    from scipy.optimize import least_squares
    tether_lengths = []
    strained_tether_lengths = []
    ypr_bridle = np.empty((flight_data.shape[0], 3))
    ypr_bridle_vk = np.empty((flight_data.shape[0], 3))
    ypr_tether = np.empty((flight_data.shape[0], 3))
    ypr_aero_force = np.empty((flight_data.shape[0], 3))
    aero_force_body = np.empty((flight_data.shape[0], 3))
    aero_force_bridle = np.empty((flight_data.shape[0], 3))
    apparent_flow_direction = np.empty((flight_data.shape[0], 2))
    kite_front_wrt_projected_velocity = np.empty(flight_data.shape[0])
    ypr_body2bridle = np.empty((flight_data.shape[0], 3))

    fig, ax_tau = plt.subplots(1, 2, sharey=True, figsize=(6.4, 3.6))
    ax_tau[1].invert_xaxis()
    wspace = .200
    plt.subplots_adjust(top=0.805, bottom=0.13, left=0.11, right=0.97, wspace=wspace)
    ax_tau[0].set_ylabel("Radial position [m]")
    ax_tau[0].set_xlabel("Up-heading position [m]")
    ax_tau[1].set_xlabel("Cross-heading position [m]")
    ax_tau[0].set_xlim([-7.5, .5])
    ax_tau[1].set_xlim([4, -4])
    for a in ax_tau: a.grid()

    mp_counter = 0
    for i, (idx, row) in enumerate(flight_data.iterrows()):
        r_kite = np.array(list(row[['rx', 'ry', 'rz']]))
        v_kite = np.array(list(row[['vx', 'vy', 'vz']]))
        args = (row['ground_tether_force'], shoot_args['n_tether_elements'], r_kite,
                list(row[['omx_opt', 'omy_opt', 'omz_opt']]), shoot_args['vwx'],
                shoot_args['separate_kcu_mass'], shoot_args['elastic_elements'])
        opt_res = least_squares(shoot, list(row[['kite_elevation', 'kite_azimuth', 'kite_distance']]), args=args, verbose=0)
        if not opt_res.success:
            print("Optimization {} failed!".format(i))
        tether_lengths.append(opt_res.x[2])
        pos_e, l_strained, dcm_b2e_i, dcm_t2e_i, dcm_fa2e_i, dcm_tau2e_i, f_aero, v_app = shoot(opt_res.x, *args, return_values=True)
        strained_tether_lengths.append(l_strained)

        ez_tau_vk = r_kite/np.linalg.norm(r_kite)
        ey_tau_vk = np.cross(ez_tau_vk, v_kite)/np.linalg.norm(np.cross(ez_tau_vk, v_kite))
        ex_tau_vk = np.cross(ey_tau_vk, ez_tau_vk)
        dcm_e2tau_vk_i = np.vstack(([ex_tau_vk], [ey_tau_vk], [ez_tau_vk]))

        pos_tau = np.zeros(pos_e.shape)
        pos_tau_vk = np.zeros(pos_e.shape)
        for j in range(pos_e.shape[0]):
            pos_tau[j, :] = dcm_tau2e_i.T.dot(pos_e[j, :])
            pos_tau_vk[j, :] = dcm_e2tau_vk_i.dot(pos_e[j, :])

        dcm_tau2b = dcm_b2e_i.T.dot(dcm_tau2e_i)
        # Note that dcm_b2e_i and dcm_tau2e_i both use the apparent wind velocity to define the positive x-axis, as such
        # the yaw should be roughly zero and it should not matter where to put the 3-rotation when unravelling.
        ypr_bridle[i, :] = unravel_euler_angles(dcm_tau2b, '321')

        dcm_tau_vk2b = dcm_b2e_i.T.dot(dcm_e2tau_vk_i.T)
        # Note that if we use 3-2-1 sequence here, we'll find the same roll as for the upper as the tau ref frame is
        # just rotated along the z-axis wrt tau_vk
        ypr_bridle_vk[i, :] = unravel_euler_angles(dcm_tau_vk2b, '213')

        dcm_tau2t = dcm_t2e_i.T.dot(dcm_tau2e_i)
        ypr_tether[i, :] = unravel_euler_angles(dcm_tau2t, '321')

        dcm_tau2fa = dcm_fa2e_i.T.dot(dcm_tau2e_i)
        ypr_aero_force[i, :] = unravel_euler_angles(dcm_tau2fa, '321')

        dcm_earth2body = rotation_matrix_earth2body(row['roll'], row['pitch'], row['yaw'])
        aero_force_body[i, :] = dcm_earth2body.dot(f_aero)
        aero_force_bridle[i, :] = dcm_b2e_i.T.dot(f_aero)
        
        v_app_body = dcm_earth2body.dot(v_app)
        apparent_flow_direction[i, 0] = -np.arctan2(v_app_body[2], v_app_body[0])
        apparent_flow_direction[i, 1] = np.arctan2(v_app_body[1], v_app_body[0])

        kite_front_wrt_projected_velocity[i] = calc_kite_kite_front_wrt_projected_velocity(r_kite, v_kite, dcm_earth2body)

        dcm_body2b = dcm_b2e_i.T.dot(dcm_earth2body.T)
        ypr_body2bridle[i, :] = unravel_euler_angles(dcm_body2b, '321')

        if i in mark_points:
            clr = 'C{}'.format(mp_counter)
            ax[0].plot3D(pos_e[:, 0], pos_e[:, 1], pos_e[:, 2], color=clr)  #, linewidth=.5, color='grey')
            ax[1].plot(pos_e[:, 0], pos_e[:, 1], color=clr, label=mp_counter+1)
            # Cross-heading is preferred opposed to cross-course as it shows a helix shape of the tether at the outside
            # of the turns, which one could interpret as caused by the centripetal acceleration, however it can be
            # to the drag.
            if mp_counter < 5:
                ax_tau[0].plot(pos_tau[:, 0], pos_tau[:, 2], color=clr, label='{}'.format(mp_counter+1))
                ax_tau[1].plot(pos_tau[:, 1], pos_tau[:, 2], color=clr)
            mp_counter += 1

        verify = False
        if verify:
            pos_e, v, a, t, fa, fnc = shoot(opt_res.x, *args, return_values=2)
            x = np.vstack((pos_e[1:, :].reshape((-1, 1)), v[1:, :].reshape((-1, 1)), [[opt_res.x[2]], [0]]))
            u = [0, *a[-1, :]]
            dp = pos_e[1:, :] - pos_e[:-1, :]
            nu = t[:, 0]/dp[:, 0]
            b = np.hstack((a[1:, :].reshape(-1), nu))

            # Checking residual when filling in system of eqs.
            c, d = dyn['f_mat'](x, u)
            eps = np.array(c@b - d)
            assert np.amax(np.abs(eps)) < 1e-6
    ax_tau[0].legend(title='Instance label', bbox_to_anchor=(.2, 1.05, 1.6+wspace, .5), loc="lower left", mode="expand",
                    borderaxespad=0, ncol=5)
    ax[1].legend(title='Instance label', bbox_to_anchor=(.0, 1.05, 1., .5), loc="lower left", mode="expand",
                    borderaxespad=0, ncol=5)

    return tether_lengths, strained_tether_lengths, ypr_bridle, ypr_bridle_vk, ypr_tether, ypr_aero_force, aero_force_body, aero_force_bridle, apparent_flow_direction, kite_front_wrt_projected_velocity, ypr_body2bridle


def find_tether_forces(flight_data, tether_lengths, shoot_args):
    assert not shoot_args['elastic_elements']
    from scipy.optimize import least_squares
    tether_forces = []
    ypr_tether_end = np.empty((flight_data.shape[0], 3))
    ypr_tether_second_last = np.empty((flight_data.shape[0], 3))
    ypr_aero_force = np.empty((flight_data.shape[0], 3))
    for i, (idx, row) in enumerate(flight_data.iterrows()):
        args = (tether_lengths[i], shoot_args['n_tether_elements'], list(row[['rx', 'ry', 'rz']]),
                list(row[['vx', 'vy', 'vz']]), list(row[['ax', 'ay', 'az']]), shoot_args['vwx'],
                shoot_args['separate_kcu_mass'], shoot_args['elastic_elements'])
        opt_res = least_squares(shoot, list(row[['kite_elevation', 'kite_azimuth', 'kite_distance']]), args=args,
                                kwargs={'find_force': True}, verbose=0)
        if not opt_res.success:
            print("Optimization failed!")
        tether_forces.append(opt_res.x[2])
        p, l_strained, dcm_t2e_i, dcm_tsl2e_i, dcm_fa2e_i, dcm_tau2e_i = shoot(opt_res.x, *args, return_values=True,
                                                                           find_force=True)
        dcm_tau2t = dcm_t2e_i.T.dot(dcm_tau2e_i)
        ypr_tether_end[i, :] = unravel_euler_angles(dcm_tau2t, '321')

        dcm_tau2tsl = dcm_tsl2e_i.T.dot(dcm_tau2e_i)
        ypr_tether_second_last[i, :] = unravel_euler_angles(dcm_tau2tsl, '321')

        dcm_tau2fa = dcm_fa2e_i.T.dot(dcm_tau2e_i)
        ypr_aero_force[i, :] = unravel_euler_angles(dcm_tau2fa, '321')
    return tether_forces, ypr_tether_end, ypr_tether_second_last, ypr_aero_force


def find_and_plot_tether_lengths(generate_sim_input=False):  #, separate_kcu_mass=True, elastic_elements=False):
    shoot_args = {
        'n_tether_elements': 30,
        'vwx': 10,
        'separate_kcu_mass': True,
        'elastic_elements': True,
    }

    from utils import read_and_transform_flight_data
    from turning_center import find_turns_for_rolling_window, determine_rigid_body_rotation
    flight_data = read_and_transform_flight_data()  # Read flight data.
    find_turns_for_rolling_window(flight_data)
    determine_rigid_body_rotation(flight_data)

    # plt.figure(figsize=(3.6, 3.6))
    fig, ax3d = plt.subplots(1, 2, figsize=(7.2, 3.6))
    plt.subplots_adjust(top=1.0, bottom=0.03, left=0.0, right=0.98, hspace=0.2, wspace=0.4)
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

    tether_lengths, strained_tether_lengths, ypr_bridle, ypr_bridle_vk, ypr_tether, ypr_aero_force, \
    aero_force_body, aero_force_bridle, flow_angles, kite_front_wrt_projected_velocity, ypr_body2bridle = find_tether_lengths(flight_data, shoot_args, ax3d)
    if shoot_args['separate_kcu_mass']:
        tether_lengths_incl_bridle = np.array(tether_lengths) + l_bridle
        strained_tether_lengths_incl_bridle = np.array(strained_tether_lengths) + l_bridle
    else:
        tether_lengths_incl_bridle = tether_lengths
        strained_tether_lengths_incl_bridle = strained_tether_lengths

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

    fig, ax_ypr = plt.subplots(2, 1, sharex=True)
    plt.suptitle("3-2-1 Euler angles between tangential\nand local ref. frames")
    if shoot_args['separate_kcu_mass']:
        ax_ypr[0].plot(flight_data.time, ypr_bridle[:, 1]*180./np.pi, label='sim bridle')
        ax_ypr[0].plot(flight_data.time, ypr_tether[:, 1]*180./np.pi, label='sim tether')
    else:
        ax_ypr[0].plot(flight_data.time, ypr_bridle[:, 1]*180./np.pi, label='sim tether')
    ax_ypr[0].plot(flight_data.time, ypr_aero_force[:, 1]*180./np.pi, label='sim aero force')
    ax_ypr[0].set_ylabel("Pitch [deg]")
    if shoot_args['separate_kcu_mass']:
        clr = ax_ypr[1].plot(flight_data.time, ypr_bridle[:, 2]*180./np.pi, label='sim bridle')[0].get_color()
        ax_ypr[1].plot(flight_data.time, ypr_bridle_vk[:, 2]*180./np.pi, '--', label='sim bridle vk', color=clr)
        ax_ypr[1].plot(flight_data.time, ypr_tether[:, 2]*180./np.pi, label='sim tether')
    else:
        ax_ypr[1].plot(flight_data.time, ypr_bridle[:, 2]*180./np.pi, label='sim tether')
    ax_ypr[1].plot(flight_data.time, ypr_aero_force[:, 2]*180./np.pi, label='sim aero force')
    ax_ypr[1].set_ylabel("Roll [deg]")
    ax_ypr[1].set_xlabel("Time [s]")

    for a in ax_ypr: a.grid()
    ax_ypr[0].plot(flight_data.time, flight_data.pitch_tau * 180. / np.pi, label='mea')
    ax_ypr[1].plot(flight_data.time, flight_data.roll_tau * 180. / np.pi, label='mea')
    ax_ypr[1].legend()
    ax_ypr[0].set_ylim([0, None])
    for a in ax_ypr: plot_flight_sections(a, flight_data)

    fig, ax = plt.subplots(4, 1, sharex=True)
    ax[0].plot(flight_data.time, np.array(strained_tether_lengths_incl_bridle)-flight_data.kite_distance, label='sag')
    # coef = np.polyfit(flight_data.time, flight_data.kite_distance, 1)
    # poly1d_fn = np.poly1d(coef)
    # delta_linear_fun = flight_data.kite_distance - poly1d_fn(flight_data.time)
    # ax[0].plot(flight_data.time, delta_linear_fun, label='delta r wrt linear')
    ax[0].legend()
    # ax[0].plot(flight_data.time, np.array(tether_lengths_incl_bridle)-flight_data.kite_distance, ':')
    ax[0].fill_between(flight_data.time, 0, 1, where=flight_data['flag_turn'], facecolor='lightsteelblue', alpha=0.5)
    ax[0].set_ylabel('Sag length [m]')
    ax[1].plot(flight_data.time, strained_tether_lengths_incl_bridle, label='strained')
    ax[1].plot(flight_data.time, tether_lengths_incl_bridle, ':', label='unstrained')
    ax[1].plot(flight_data.time, flight_data.kite_distance, '-.', label='radial position')
    # bias = np.mean(tether_lengths_incl_bridle - flight_data.ground_tether_length)
    # print("Tether length bias", bias)
    # ax[1].plot(flight_data.time, flight_data.ground_tether_length + bias, '-.', label='mea')
    ax[1].legend()
    ax[1].set_ylabel('Tether length [m]')
    ax[2].plot(flight_data.time[:-1], np.diff(tether_lengths)/.1)
    ax[2].plot(flight_data.time, flight_data.ground_tether_reelout_speed, ':')
    ax[2].set_ylabel('Tether speed [m/s]')

    ax[3].set_ylabel('Tether acceleration [m/s**2]')

    if not shoot_args['elastic_elements']:
        tether_length_and_speed, tether_acceleration = match_tether_length_and_speed(tether_lengths_incl_bridle, flight_data.ground_tether_reelout_speed, flight_data.kite_distance)
        print("Start tether speed", tether_length_and_speed[0, 1])
        ax[0].plot(flight_data.time, tether_length_and_speed[:, 0]-flight_data.kite_distance, '--')
        ax[1].plot(flight_data.time, tether_length_and_speed[:, 0], '--')
        ax[2].plot(flight_data.time, tether_length_and_speed[:, 1], '--')
        ax[3].plot(flight_data.time[:-1], tether_acceleration)

        if generate_sim_input:
            if shoot_args['separate_kcu_mass']:
                tether_length_and_speed[:, 0] = tether_length_and_speed[:, 0]-l_bridle
            opt_controls = np.hstack((tether_acceleration, [np.nan])).reshape((-1, 1))
            save_array = np.hstack((tether_length_and_speed, opt_controls))
            np.save('tether_states.npy', save_array)
            if shoot_args['separate_kcu_mass']:
                np.save('ypr_bridle_rigid_body_rotation.npy', ypr_bridle)

        # tether_forces, ypr_tether_end2, ypr_tether_second_last2, ypr_aero_force2 = find_tether_forces(flight_data, tether_length_and_speed[:, 0], shoot_args)
        # ax_ypr[1].plot(flight_data.time, ypr_tether_end2[:, 2]*180./np.pi, 'k', label='sim bridle2')

    plt.show()


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

    # # Initial guesses
    # opti.set_initial(states, df['ground_tether_reelout_speed'].values)
    # opti.set_initial(controls, np.diff(df['ground_tether_reelout_speed'].values)/.1)

    opti.minimize(ca.sumsqr(states[:, 0] - np.array(tether_lengths)) + ca.sumsqr(states[:, 1] - np.array(tether_speeds)))
    opti.subject_to(opti.bounded(-.1, controls[1:]-controls[:-1], .1))
    opti.subject_to(states[:, 0] - radius > .01)

    # solve optimization problem
    opti.solver('ipopt')

    sol = opti.solve()

    return sol.value(states), sol.value(controls)


if __name__ == "__main__":
    find_and_plot_tether_lengths()
