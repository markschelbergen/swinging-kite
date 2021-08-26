import numpy as np
from utils import unravel_euler_angles, plot_flight_sections


def plot_vector(p0, v, ax, scale_vector=1, color='g', label=None):
    p1 = p0 + v * scale_vector
    vector = np.vstack(([p0], [p1])).T
    ax.plot3D(vector[0], vector[1], vector[2], color=color, label=label)


rho = 1.225
g = 9.81
d_t = .01
rho_t = 724.
cd_t = 1.1

m_kite = 11
m_kcu = 9
l_bridle = 11.5


def shoot(x, n_tether_elements, r_kite, v_kite, tension_ground, vwx, separate_kcu_mass=False, ax_plot_forces=False, return_positions=False):
    # Currently neglecting radial velocity of kite.
    beta_n, phi_n, tether_length = x
    om = np.cross(r_kite, v_kite)/np.linalg.norm(r_kite)**2
    l_s = tether_length/n_tether_elements
    m_s = np.pi*d_t**2/4 * l_s * rho_t

    n_elements = n_tether_elements
    if separate_kcu_mass:
        n_elements += 1

    tensions = np.zeros((n_elements, 3))
    tensions[0, 0] = np.cos(beta_n)*np.cos(phi_n)*tension_ground
    tensions[0, 1] = np.sin(phi_n)*tension_ground
    tensions[0, 2] = np.sin(beta_n)*np.cos(phi_n)*tension_ground

    positions = np.zeros((n_elements+1, 3))
    positions[1, 0] = np.cos(beta_n)*np.cos(phi_n)*l_s
    positions[1, 1] = np.sin(phi_n)*l_s
    positions[1, 2] = np.sin(beta_n)*np.cos(phi_n)*l_s

    er = r_kite/np.linalg.norm(r_kite)
    vr = np.dot(v_kite, er)*er  # Radial velocity of 'rigid body'

    for j in range(n_elements):
        last_element = j == n_elements - 1
        kcu_element = separate_kcu_mass and j == n_elements - 2

        vj = np.cross(om, positions[j+1, :]) #+ vr  # TODO: only angular rotation considered
        aj = np.cross(om, np.cross(om, positions[j+1, :]))
        ej = (positions[j+1, :] - positions[j, :])/l_s  # Axial direction of tether element

        vaj = vj - np.array([vwx, 0, 0])  # Apparent wind velocity
        vajp = np.dot(vaj, ej)*ej  # Parallel to tether element
        vajn = vaj - vajp  # Perpendicular to tether element

        if separate_kcu_mass and last_element:
            dj = 0  # TODO: add bridle drag
        else:
            dj = -.5*rho*l_s*d_t*cd_t*np.linalg.norm(vajn)*vajn

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
                dj += -.5*rho*1.6*1.*np.linalg.norm(vajn)*vajn
            else:
                point_mass = m_s

        fgj = np.array([0, 0, -point_mass*g])
        resultant_force = point_mass*aj + tensions[j, :] - dj - fgj
        if not last_element:
            tensions[j+1, :] = resultant_force
        else:
            aerodynamic_force = resultant_force

        if ax_plot_forces:
            plot_vector(positions[j+1, :], dj, ax_plot_forces, color='r')
            plot_vector(positions[j+1, :], fgj, ax_plot_forces, color='k')
            plot_vector(positions[j+1, :], -point_mass*aj, ax_plot_forces, color='m')
            plot_vector(positions[j+1, :], -tensions[j, :], ax_plot_forces, color='g')
            plot_vector(positions[j+1, :], resultant_force, ax_plot_forces, color='b')

        if kcu_element:
            positions[j+2, :] = positions[j+1, :] + tensions[j+1, :]/np.linalg.norm(tensions[j+1, :]) * l_bridle
        elif not last_element:
            positions[j+2, :] = positions[j+1, :] + tensions[j+1, :]/np.linalg.norm(tensions[j+1, :]) * l_s

    if return_positions:
        va = v_kite - np.array([vwx, 0, 0])

        ez_last_elem = tensions[-1, :]/np.linalg.norm(tensions[-1, :])
        ey_last_elem = np.cross(ez_last_elem, va)/np.linalg.norm(np.cross(ez_last_elem, va))
        ex_last_elem = np.cross(ey_last_elem, ez_last_elem)
        dcm_last_elem = np.vstack(([ex_last_elem], [ey_last_elem], [ez_last_elem])).T

        ez_2nd_last_elem = tensions[-2, :]/np.linalg.norm(tensions[-2, :])
        ey_2nd_last_elem = np.cross(ez_2nd_last_elem, va)/np.linalg.norm(np.cross(ez_2nd_last_elem, va))
        ex_2nd_last_elem = np.cross(ey_2nd_last_elem, ez_2nd_last_elem)
        dcm_2nd_last_elem = np.vstack(([ex_2nd_last_elem], [ey_2nd_last_elem], [ez_2nd_last_elem])).T

        ez_f_aero = aerodynamic_force/np.linalg.norm(aerodynamic_force)
        ey_f_aero = np.cross(ez_f_aero, va)/np.linalg.norm(np.cross(ez_f_aero, va))
        ex_f_aero = np.cross(ey_f_aero, ez_f_aero)
        dcm_f_aero = np.vstack(([ex_f_aero], [ey_f_aero], [ez_f_aero])).T

        ez_tau = r_kite/np.linalg.norm(r_kite)
        ey_tau = np.cross(ez_tau, va)/np.linalg.norm(np.cross(ez_tau, va))
        ex_tau = np.cross(ey_tau, ez_tau)
        dcm_tau = np.vstack(([ex_tau], [ey_tau], [ez_tau])).T

        return positions, dcm_last_elem, dcm_2nd_last_elem, dcm_f_aero, dcm_tau
    else:
        return positions[-1, :] - r_kite


def find_tether_length():
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    from scipy.optimize import least_squares

    args = (10, [200, 0, 100], [2, 20, 1], 1000, 9)
    opt_res = least_squares(shoot, (20*np.pi/180., -15*np.pi/180., 250), args=args, verbose=2)
    print("Resulting tether length:", opt_res.x[2])
    p = shoot(opt_res.x, *args, return_positions=True)[0]

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


def find_tether_lengths_flight_data(generate_sim_input=False, separate_kcu_mass=True):
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    from scipy.optimize import least_squares
    from utils import read_and_transform_flight_data
    # Difference as low as 0.009 m up to 0.31 m
    vwx = 9
    n_tether_elements = 30
    flight_data = read_and_transform_flight_data()  # Read flight data.
    if generate_sim_input:
        # Using inferred position compatible with integrator such that the tether acceleration can be used as input for
        # the simulation.
        flight_data[['rx', 'ry', 'rz']] = np.load('kite_states.npy')[:, :3]
        flight_data['radius'] = np.sum(flight_data[['rx', 'ry', 'rz']].values**2, axis=1)**.5

    plt.figure(figsize=(8, 6))
    ax3d = plt.axes(projection='3d')
    ax3d.plot3D(flight_data.rx, flight_data.ry, flight_data.rz)
    ax3d.set_xlim([0, 250])
    ax3d.set_ylim([-125, 125])
    ax3d.set_zlim([0, 250])

    tether_lengths = []
    ypr_tether_end = np.empty((flight_data.shape[0], 3))
    ypr_tether_second_last = np.empty((flight_data.shape[0], 3))
    ypr_aero_force = np.empty((flight_data.shape[0], 3))
    for i, (idx, row) in enumerate(flight_data.iterrows()):
        args = (n_tether_elements, list(row[['rx', 'ry', 'rz']]), list(row[['vx', 'vy', 'vz']]),
                row['ground_tether_force'], vwx, separate_kcu_mass)
        opt_res = least_squares(shoot, list(row[['kite_elevation', 'kite_azimuth', 'radius']]), args=args, verbose=0)
        if not opt_res.success:
            print("Optimization failed!")
        tether_lengths.append(opt_res.x[2])
        if separate_kcu_mass:
            tether_lengths[-1] = tether_lengths[-1] + l_bridle
        p, rm_t2e_i, rm_tsl2e_i, rm_fa2e_i, rm_tau2e_i = shoot(opt_res.x, *args, return_positions=True)

        rm_tau2t = rm_t2e_i.T.dot(rm_tau2e_i)
        ypr_tether_end[i, :] = unravel_euler_angles(rm_tau2t, '321')

        rm_tau2tsl = rm_tsl2e_i.T.dot(rm_tau2e_i)
        ypr_tether_second_last[i, :] = unravel_euler_angles(rm_tau2tsl, '321')

        rm_tau2fa = rm_fa2e_i.T.dot(rm_tau2e_i)
        ypr_aero_force[i, :] = unravel_euler_angles(rm_tau2fa, '321')

        if i % 25 == 0:
            ax3d.plot3D(p[:, 0], p[:, 1], p[:, 2], linewidth=.5, color='grey')

    flight_data['calculated_tether_length'] = tether_lengths

    fig, ax_ypr = plt.subplots(3, 1, sharex=True)
    plt.suptitle("3-2-1 Euler angles between tangential\nand last tether element ref. frame")
    ax_ypr[0].plot(flight_data.time, ypr_tether_end[:, 0]*180./np.pi, label='sim')
    ax_ypr[0].set_ylabel("Yaw [deg]")
    if separate_kcu_mass:
        ax_ypr[1].plot(flight_data.time, ypr_tether_end[:, 1]*180./np.pi, label='sim bridle')
        ax_ypr[1].plot(flight_data.time, ypr_tether_second_last[:, 1]*180./np.pi, label='sim tether')
    else:
        ax_ypr[1].plot(flight_data.time, ypr_tether_end[:, 1]*180./np.pi, label='sim tether')
    ax_ypr[1].plot(flight_data.time, ypr_aero_force[:, 1]*180./np.pi, label='sim fa')
    ax_ypr[1].set_ylabel("Pitch [deg]")
    if separate_kcu_mass:
        ax_ypr[2].plot(flight_data.time, ypr_tether_end[:, 2]*180./np.pi, label='sim bridle')
        ax_ypr[2].plot(flight_data.time, ypr_tether_second_last[:, 2]*180./np.pi, label='sim tether')
    else:
        ax_ypr[2].plot(flight_data.time, ypr_tether_end[:, 2]*180./np.pi, label='sim tether')
    ax_ypr[2].plot(flight_data.time, ypr_aero_force[:, 2]*180./np.pi, label='sim fa')
    ax_ypr[2].set_ylabel("Roll [deg]")
    ax_ypr[2].set_xlabel("Time [s]")

    for a in ax_ypr: a.grid()
    ax_ypr[1].plot(flight_data.time, flight_data.pitch_tau * 180. / np.pi, label='mea')
    ax_ypr[2].plot(flight_data.time, flight_data.roll_tau * 180. / np.pi, label='mea')
    ax_ypr[2].legend()
    ax_ypr[1].set_ylim([0, None])
    for a in ax_ypr: plot_flight_sections(a, flight_data)

    opt_states, opt_controls = match_measured_tether_speed(tether_lengths, flight_data.radius)

    fig, ax = plt.subplots(4, 1, sharex=True)
    ax[0].plot(flight_data.time, flight_data.calculated_tether_length-flight_data.radius)
    ax[0].plot(flight_data.time, opt_states[:, 0]-flight_data.radius, '--')
    ax[1].plot(flight_data.time, tether_lengths)
    ax[1].plot(flight_data.time, opt_states[:, 0], '--')
    ax[2].plot(flight_data.time[:-1], np.diff(tether_lengths)/.1)
    ax[2].plot(flight_data.time, opt_states[:, 1], '--')
    ax[3].plot(flight_data.time[:-1], opt_controls)

    if generate_sim_input:
        opt_controls = np.hstack((opt_controls, [np.nan])).reshape((-1, 1))
        np.save('tether_states.npy', np.hstack((opt_states, opt_controls)))

    plt.show()


def match_measured_tether_speed(tether_lengths, radius):
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

    opti.minimize(ca.sumsqr(states[:, 0] - np.array(tether_lengths)))
    opti.subject_to(opti.bounded(-.1, controls[1:]-controls[:-1], .1))
    opti.subject_to(states[:, 0] - radius > .01)

    # solve optimization problem
    opti.solver('ipopt')

    sol = opti.solve()

    return sol.value(states), sol.value(controls)


if __name__ == "__main__":
    find_tether_lengths_flight_data()
