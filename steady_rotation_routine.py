import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import pickle

from utils import unravel_euler_angles, plot_flight_sections, rotation_matrix_earth2body, \
    read_and_transform_flight_data, add_panel_labels
from system_properties import *
from dynamic_model import derive_tether_model_kcu_williams
from turning_center import mark_points, determine_steady_rotation


def plot_vector(p0, v, ax, scale_vector=.03, color='g', label=None):
    p1 = p0 + v * scale_vector
    vector = np.vstack(([p0], [p1])).T
    ax.plot3D(vector[0], vector[1], vector[2], color=color, label=label)


def get_tether_end_position(x, config):
    set_parameter = config['set_parameter']
    n_tether_elements = config['n_tether_elements']
    r_kite = config['r_kite']
    omega = config['omega']
    v_kite_radial = config.get('v_kite_radial', 0)
    separate_kcu_mass = config.get('separate_kcu_mass', False)
    elastic_elements = config.get('elastic_elements', True)
    ax_plot_forces = config.get('ax_plot_forces', False)
    return_values = config.get('return_values', False)
    find_force = config.get('find_force', False)
    drag_perpendicular = config.get('drag_perpendicular', False)

    if find_force:
        beta_n, phi_n, tension_ground = x
        tether_length_wo_bridle = set_parameter
    else:
        beta_n, phi_n, tether_length_wo_bridle = x
        tension_ground = set_parameter

    l_unstrained = tether_length_wo_bridle/n_tether_elements
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

    calc_tether_length_wo_bridle = l_s  # Stretched
    for j in range(n_elements):  # Iterate over point masses.
        last_element = j == n_elements - 1
        kcu_element = separate_kcu_mass and j == n_elements - 2

        # Determine kinematics at point mass j.
        vj = np.cross(omega, positions[j+1, :]) + v_kite_radial
        velocities[j+1, :] = vj
        aj = np.cross(omega, np.cross(omega, positions[j+1, :]))
        accelerations[j+1, :] = aj

        # Determine flow at point mass j.
        vaj = vj - np.array([vwx, 0, 0])  # Apparent wind velocity
        if not drag_perpendicular:
            vaj_sq = np.linalg.norm(vaj)*vaj
        else:
            delta_p = positions[j+1, :] - positions[j, :]
            ej = delta_p / np.linalg.norm(delta_p)  # Axial direction of tether element
            vajp = np.dot(vaj, ej) * ej  # Parallel to tether element
            vajn = vaj - vajp  # Perpendicular to tether element
            vaj_sq = np.linalg.norm(vajn)*vajn
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
            labels = ['resultant', 'drag', 'weight']  #, 'last tension', 'next tension']
            clrs = ['m', 'r', 'k', 'g', 'b']
            for f, lbl, clr in zip(forces, labels, clrs):
                # print("{} = {:.2f} N".format(lbl, np.linalg.norm(f)))
                plot_vector(positions[j+1, :], f*50, ax_plot_forces, color=clr)
        elif ax_plot_forces:
            ax_plot_forces.set_xlim([0, 300])
            ax_plot_forces.set_ylim([-150, 150])
            ax_plot_forces.set_zlim([0, 300])
            ax_plot_forces.set_box_aspect([1.0, 1.0, 1.0])
            ax_plot_forces.plot(positions[:, 0], positions[:, 1], positions[:, 2])

        # Derive position of next point mass from former tension
        if kcu_element:
            positions[j+2, :] = positions[j+1, :] + tensions[j+1, :]/np.linalg.norm(tensions[j+1, :]) * l_bridle
        elif not last_element:
            if elastic_elements:
                l_s = (np.linalg.norm(tensions[j+1, :])/tether_stiffness+1)*l_unstrained
            else:
                l_s = l_unstrained
            calc_tether_length_wo_bridle += l_s
            positions[j+2, :] = positions[j+1, :] + tensions[j+1, :]/np.linalg.norm(tensions[j+1, :]) * l_s

    if return_values == 2:
        return positions, velocities, accelerations, tensions, aerodynamic_force, non_conservative_forces
    elif return_values:
        va = vj - np.array([vwx, 0, 0])  # All y-axes are defined perpendicular to apparent wind velocity.

        ez_bridle = tensions[-1, :]/np.linalg.norm(tensions[-1, :])
        ey_bridle = np.cross(ez_bridle, va)/np.linalg.norm(np.cross(ez_bridle, va))
        ex_bridle = np.cross(ey_bridle, ez_bridle)
        dcm_b2w = np.vstack(([ex_bridle], [ey_bridle], [ez_bridle])).T

        ez_tether = tensions[-2, :]/np.linalg.norm(tensions[-2, :])
        ey_tether = np.cross(ez_tether, va)/np.linalg.norm(np.cross(ez_tether, va))
        ex_tether = np.cross(ey_tether, ez_tether)
        dcm_t2w = np.vstack(([ex_tether], [ey_tether], [ez_tether])).T

        ez_f_aero = aerodynamic_force/np.linalg.norm(aerodynamic_force)
        ey_f_aero = np.cross(ez_f_aero, va)/np.linalg.norm(np.cross(ez_f_aero, va))
        ex_f_aero = np.cross(ey_f_aero, ez_f_aero)
        dcm_fa2w = np.vstack(([ex_f_aero], [ey_f_aero], [ez_f_aero])).T

        ez_tau = r_kite/np.linalg.norm(r_kite)
        ey_tau = np.cross(ez_tau, va)/np.linalg.norm(np.cross(ez_tau, va))
        ex_tau = np.cross(ey_tau, ez_tau)
        dcm_tau2w = np.vstack(([ex_tau], [ey_tau], [ez_tau])).T

        return positions, calc_tether_length_wo_bridle, dcm_b2w, dcm_t2w, dcm_fa2w, \
               dcm_tau2w, aerodynamic_force, va
    else:
        return positions[-1, :] - r_kite


def find_tether_lengths(flight_data, config, plot=False, plot_instances=mark_points):
    phi_upwind_direction = -flight_data.loc[flight_data.index[0], 'est_upwind_direction']-np.pi/2.
    from scipy.optimize import least_squares

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

    n_elements = config['n_tether_elements']
    if config['separate_kcu_mass']:
        n_elements += 1

    pos_tau = np.zeros((n_rows, n_elements+1, 3))
    pos_tau_vk = np.zeros((n_rows, n_elements+1, 3))
    orientation_table = []

    mp_counter = 0
    for i, (idx, row) in enumerate(flight_data.iterrows()):
        # Find consistent tether shape
        r_kite = np.array(list(row[['rx', 'ry', 'rz']]))
        v_kite = np.array(list(row[['vx', 'vy', 'vz']]))
        gtep_config = {
            'n_tether_elements': config['n_tether_elements'],
            'separate_kcu_mass': config['separate_kcu_mass'],
            'elastic_elements': config['elastic_elements'],
            'set_parameter': row['ground_tether_force'],
            'r_kite': r_kite,
            'v_kite_radial': np.dot(v_kite, r_kite)/np.linalg.norm(r_kite)**2 * r_kite,
            'omega': list(row[['omx', 'omy', 'omz']]),
            'drag_perpendicular': True,
        }
        opt_res = least_squares(get_tether_end_position, list(row[['kite_elevation', 'kite_azimuth', 'kite_distance']]), args=(gtep_config,), verbose=0)
        if not opt_res.success:
            print("Optimization {} failed!".format(i))
        tether_lengths.append(opt_res.x[2])
        gtep_config['return_values'] = True

        pos_w, l_strained, dcm_b2w_i, dcm_t2w_i, dcm_fa2w_i, dcm_tau2w_i, f_aero, v_app = get_tether_end_position(opt_res.x, gtep_config)
        strained_tether_lengths.append(l_strained)

        # Determine orientation of vertical and centrifugal force in tau reference frame
        if i in plot_instances:
            om = np.array(list(row[['omx', 'omy', 'omz']]))
            a_c = np.cross(om, np.cross(om, r_kite))
            e_cf = -a_c/np.linalg.norm(a_c)
            orientation_table.append(np.hstack((dcm_tau2w_i.T.dot([0, 0, -1])[:2], dcm_tau2w_i.T.dot(e_cf)[:2])))

        ez_tau_vk = r_kite/np.linalg.norm(r_kite)
        ey_tau_vk = np.cross(ez_tau_vk, v_kite)/np.linalg.norm(np.cross(ez_tau_vk, v_kite))
        ex_tau_vk = np.cross(ey_tau_vk, ez_tau_vk)
        dcm_w2tau_vk_i = np.vstack(([ex_tau_vk], [ey_tau_vk], [ez_tau_vk]))

        # dcm = rotation_matrix_earth2sphere(row['kite_azimuth'], row['kite_elevation'])  # Could be used to show
        # cross-axial displacement wrt to elevation and aziumth instead of wrt heading
        for j in range(pos_w.shape[0]):
            # dcm.dot(pos_e[j, :])
            pos_tau[i, j, :] = dcm_tau2w_i.T.dot(pos_w[j, :])
            pos_tau_vk[i, j, :] = dcm_w2tau_vk_i.dot(pos_w[j, :])

        dcm_tau2b = dcm_b2w_i.T.dot(dcm_tau2w_i)
        # Note that dcm_b2e_i and dcm_tau2w_i both use the apparent wind velocity to define the positive x-axis, as such
        # the yaw should be roughly zero and it should not matter where to put the 3-rotation when unravelling.
        ypr_bridle[i, :] = unravel_euler_angles(dcm_tau2b, '321')

        dcm_tau_vk2b = dcm_b2w_i.T.dot(dcm_w2tau_vk_i.T)
        # Note that if we use 3-2-1 sequence here, we'll find the same roll as for the upper as the tau ref frame is
        # just rotated along the z-axis wrt tau_vk
        ypr_bridle_vk[i, :] = unravel_euler_angles(dcm_tau_vk2b, '213')

        dcm_tau2t = dcm_t2w_i.T.dot(dcm_tau2w_i)
        ypr_tether[i, :] = unravel_euler_angles(dcm_tau2t, '321')

        dcm_tau2fa = dcm_fa2w_i.T.dot(dcm_tau2w_i)
        ypr_aero_force[i, :] = unravel_euler_angles(dcm_tau2fa, '321')

        if plot and i in plot_instances:
            clr = 'C{}'.format(mp_counter)
            ax3d[0].plot3D(pos_w[:, 0], pos_w[:, 1], pos_w[:, 2], color=clr)  #, linewidth=.5, color='grey')
            ax3d[1].plot(pos_w[:, 0], pos_w[:, 1], color=clr, label=mp_counter + 1)
            # Cross-heading is preferred opposed to cross-course as it shows a helix shape of the tether at the outside
            # of the turns, which one could interpret as caused by the centripetal acceleration, however it can be
            # to the drag.
            mp_counter += 1

        verify = False
        if verify:
            pos_w, v, a, t, fa, fnc = get_tether_end_position(opt_res.x, *args, return_values=2)
            x = np.vstack((pos_w[1:, :].reshape((-1, 1)), v[1:, :].reshape((-1, 1)), [[opt_res.x[2]], [0]]))
            u = [0, *a[-1, :]]
            dp = pos_w[1:, :] - pos_w[:-1, :]
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

    # for row in np.array(orientation_table):
    #     print(" & ".join([f"{val:.2f}" for val in row]))

    return tether_lengths, strained_tether_lengths, ypr_bridle, ypr_bridle_vk, ypr_tether, ypr_aero_force, pos_tau


def find_tether_forces(flight_data, config, tether_lengths):
    assert len(tether_lengths) == flight_data.shape[0], f"Mismatch: {len(tether_lengths)}/{flight_data.shape[0]}"
    from scipy.optimize import least_squares

    tether_forces = []
    for i, (idx, row) in enumerate(flight_data.iterrows()):
        # Find consistent tether shape
        r_kite = np.array(list(row[['rx', 'ry', 'rz']]))
        v_kite = np.array(list(row[['vx', 'vy', 'vz']]))
        gtep_config = {
            'n_tether_elements': config['n_tether_elements'],
            'separate_kcu_mass': config['separate_kcu_mass'],
            'elastic_elements': config['elastic_elements'],
            'set_parameter': tether_lengths[i],
            'r_kite': r_kite,
            'v_kite_radial': np.dot(v_kite, r_kite)/np.linalg.norm(r_kite)**2 * r_kite,
            'omega': list(row[['omx', 'omy', 'omz']]),
            'drag_perpendicular': True,
            'find_force': True,
        }
        opt_res = least_squares(get_tether_end_position, list(row[['kite_elevation', 'kite_azimuth', 'kite_distance']]), args=(gtep_config,), verbose=0)
        if not opt_res.success:
            print("Optimization {} failed!".format(i))
        tether_forces.append(opt_res.x[2])

    return tether_forces


def plot_tether_element_pitch(flight_data, ypr_bridle, ax):
    ax.grid()
    ax.plot(flight_data.time, flight_data.pitch0_tau * 180. / np.pi, label='Sensor 0')
    ax.plot(flight_data.time, flight_data.pitch1_tau * 180. / np.pi, label='Sensor 1')
    ax.plot(flight_data.time, ypr_bridle[:, 1]*180./np.pi, label='Steady-rotation N=30')
    ax.set_xlim([flight_data.iloc[0]['time'], flight_data.iloc[-1]['time']])
    plot_flight_sections(ax, flight_data)


def plot_tether_element_attitudes(flight_data, ypr_aero_force, ypr_bridle, ypr_tether, separate_kcu_mass, mark_instances=None):
    plot_yaw = True
    if plot_yaw:
        n_rows = 3
    else:
        n_rows = 2
    fig, ax_ypr = plt.subplots(n_rows, 1, sharex=True)
    plt.suptitle("3-2-1 Euler angles between tangential\nand local ref. frames")
    if separate_kcu_mass:
        clr = ax_ypr[0].plot(flight_data.time, ypr_bridle[:, 1]*180./np.pi, label='Bridle')[0].get_color()
        if mark_instances is not None:
            ax_ypr[0].plot(flight_data.time[flight_data.index[mark_instances]], ypr_bridle[mark_instances, [1]*len(mark_instances)]*180./np.pi, 's', color=clr)
        ax_ypr[0].plot(flight_data.time, ypr_tether[:, 1]*180./np.pi, label='Tether')
    else:
        ax_ypr[0].plot(flight_data.time, ypr_bridle[:, 1]*180./np.pi, label='Tether')
    ax_ypr[0].set_ylabel("Pitch [deg]")

    ax_ypr[1].plot(flight_data.time, ypr_aero_force[:, 2]*180./np.pi, label='Aero force')
    if separate_kcu_mass:
        ax_ypr[1].plot(flight_data.time, ypr_bridle[:, 2]*180./np.pi, label='Bridle')[0].get_color()
        ax_ypr[1].plot(flight_data.time, ypr_tether[:, 2]*180./np.pi, label='Tether')
    else:
        ax_ypr[1].plot(flight_data.time, ypr_bridle[:, 2]*180./np.pi, label='Tether')
    ax_ypr[1].set_ylabel("Roll [deg]")

    if plot_yaw:
        ax_ypr[2].plot(flight_data.time, ypr_bridle[:, 0]*180./np.pi, label='Bridle')
        ax_ypr[2].set_ylabel("Yaw [deg]")

    ax_ypr[-1].set_xlabel("Time [s]")

    for a in ax_ypr: a.grid()
    ax_ypr[0].plot(flight_data.time, flight_data.pitch0_tau * 180. / np.pi, label='Sensor 0')[0]
    ax_ypr[0].plot(flight_data.time, flight_data.pitch1_tau * 180. / np.pi, label='Sensor 1')[0]
    ax_ypr[0].plot(flight_data.time, flight_data.pitch_tau * 180. / np.pi, label='Sensor avg')[0]
    ax_ypr[0].legend()
    ax_ypr[1].plot(flight_data.time, flight_data.roll_tau * 180. / np.pi, label='Sensor avg')
    ax_ypr[1].legend()
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

    # ax[0, 1].plot(flight_data.time[:-1], np.diff(strained_tether_lengths)/.1, linewidth=.5)
    ax[0, 1].plot(flight_data.time, flight_data.ground_tether_reelout_speed)
    ax[0, 1].set_ylabel('Tether speed [m/s]')
    # ax[0, 1].set_ylim([0, 2.5])

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


def find_and_plot_tether_lengths(n_tether_elements=30, i_cycle=None, ax=None, config=None, plot_interval=(29.9, 51.2), plot_3d_tether_shapes=False):
    if config is None:
        config = {
            'n_tether_elements': n_tether_elements,
            'separate_kcu_mass': True,
            'elastic_elements': False,
            'make_kinematics_consistent': True,
        }

    flight_data = read_and_transform_flight_data(config['make_kinematics_consistent'], i_cycle)  # Read flight data.
    if i_cycle is not None:
        flight_data.loc[flight_data['phase'] >= 3, 'azimuth_turn_center'] = np.nan
        flight_data.loc[flight_data['phase'] >= 3, 'elevation_turn_center'] = np.nan
        flight_data.loc[flight_data['phase'] >= 3, 'flag_turn'] = False
    determine_steady_rotation(flight_data)

    plot_interval_idx = (
        (flight_data['time'] == plot_interval[0]).idxmax(),
        (flight_data['time'] == plot_interval[1]).idxmax()
    )
    plot_interval_irow = (
        plot_interval_idx[0] - flight_data.index[0],
        plot_interval_idx[1] - flight_data.index[0],
    )
    plot_instances = [i+plot_interval_irow[0] for i in mark_points]

    tether_lengths, strained_tether_lengths, ypr_bridle, ypr_bridle_vk, ypr_tether, ypr_aero_force, pos_tau = \
        find_tether_lengths(flight_data, config, plot=plot_3d_tether_shapes, plot_instances=plot_instances)

    # plt.figure()
    # plt.ylabel('Tether slack [m]')
    # plt.xlabel('Time [s]')
    # plt.plot(flight_data['time'], tether_lengths-flight_data['l'], label='unstrained')
    # plt.plot(flight_data['time'], strained_tether_lengths-flight_data['l'], label='strained')
    # plt.plot(flight_data['time'], strained_tether_lengths-tether_lengths, label='diff')
    # plt.legend()
    # print(f"Mean tether slack {np.mean(strained_tether_lengths-flight_data['l']):.3f}")
    # plt.xlim([flight_data.loc[plot_interval_idx[0], 'time'], flight_data.loc[plot_interval_idx[1], 'time']])

    if ax is not None:
        assert config['separate_kcu_mass'], "Analysis only used with bridle element"
        plot_tether_element_pitch(flight_data, ypr_bridle, ax)
    else:
        if config['separate_kcu_mass'] and not config['elastic_elements'] and i_cycle is None:
            res = {
                'strained_tether_lengths': strained_tether_lengths,
                'pitch_bridle': ypr_bridle[:, 1],
                'roll_bridle': ypr_bridle[:, 2],
                'offaxial_tether_shape': pos_tau,
            }
            with open("results/steady_rotation_results{}.pickle".format(config['n_tether_elements']), 'wb') as f:
                pickle.dump(res, f)

def find_and_plot_tether_forces(tether_lengths):
    n_tether_elements = 30
    i_cycle = None
    config = {
        'n_tether_elements': n_tether_elements,
        'separate_kcu_mass': True,
        'elastic_elements': False,
        'make_kinematics_consistent': True,
    }
    flight_data = read_and_transform_flight_data(config['make_kinematics_consistent'], i_cycle)  # Read flight data.
    determine_steady_rotation(flight_data)
    tether_forces = find_tether_forces(flight_data, config, tether_lengths)
    plt.gca().plot(flight_data['time'], np.array(tether_forces)*1e-3, '--')


def match_tether_length_and_speed(tether_lengths, radius, tether_speeds=None):
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

    if tether_speeds is None:
        obj = ca.sumsqr(states[:, 0] - np.array(tether_lengths))
        opti.subject_to(states[:, 0] - radius > .01)
        # opti.subject_to(opti.bounded(-.3, controls[1:]-controls[:-1], .3))
    else:
        obj1 = ca.sumsqr(states[:, 1] - np.array(tether_speeds))
        obj2 = ca.sumsqr(states[:, 0] - radius)
        obj = obj1 + obj2
        # opti.subject_to(states[0, 0] - radius[0] == 0)

    control_steps = controls[1:, :]-controls[:-1, :]
    weight_control_steps = 1/15  # sumsqr of control steps resulting with given weights: {1/15: 3606, 1/30: 5526, 1/100: 12315, 0: 21295}
    obj += ca.sumsqr(weight_control_steps*control_steps)
    opti.minimize(obj)

    # solve optimization problem
    opti.solver('ipopt')

    sol = opti.solve()
    print(sol.value(obj))
    # print(sol.value(ca.sumsqr(.1*(states[:, 1] - np.array(tether_speeds)))))
    # print(sol.value(ca.sumsqr(weight_control_steps*control_steps)))

    return sol.value(states), sol.value(controls)


def plot_pitch_multi_cycles():
    fig, ax = plt.subplots(5, 2, sharey=True, figsize=[6.4, 6.4])
    left, right, wspace = 0.105, 0.985, 0.085
    plt.subplots_adjust(top=0.945, bottom=0.085, left=left, right=right, hspace=0.36, wspace=0.085)
    fig.supxlabel('Time [s]', x=(left+right)/2)
    fig.supylabel("Pitch [$^\circ$]")
    for ic, a in zip(list(range(65, 75)), ax.reshape(-1)):
        find_and_plot_tether_lengths(30, i_cycle=ic, ax=a)
        a.text(.05, .75, "Cycle {}".format(ic), transform=a.transAxes)
    ax[0, 0].legend(bbox_to_anchor=(.2, 1.07, 1.6+wspace, .5), loc="lower left", mode="expand",
                       borderaxespad=0, ncol=3)
    plt.show()


if __name__ == "__main__":
    find_and_plot_tether_lengths(1)  # Generates results single element tether
    find_and_plot_tether_lengths(30, plot_3d_tether_shapes=True)  # Generates results multi-element tether and plots figure 9
    plot_pitch_multi_cycles()  # Plots figure 15
    plt.show()
