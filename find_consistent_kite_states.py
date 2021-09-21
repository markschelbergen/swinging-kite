import casadi as ca
import numpy as np


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
    from utils import tranform_to_wind_rf
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
    mea_states = df[['rx', 'ry', 'rz', 'vx', 'vy', 'vz']].values
    opti.set_initial(states, mea_states)

    # accelerations = np.vstack([[np.gradient(df['vx'])/.1], [np.gradient(df['vy'])/.1], [np.gradient(df['vz'])/.1]]).T
    accelerations = df[['kite_1_ax', 'kite_1_ay', 'kite_1_az']].values
    opti.set_initial(controls, accelerations[:-1, :])

    weights = np.ones(mea_states.shape)
    # weights[:, :3] = 3

    opti.minimize(ca.sumsqr(weights*(states - mea_states)))

    # solve optimization problem
    opti.solver('ipopt')

    sol = opti.solve()

    states_sol = sol.value(states)
    controls_sol = sol.value(controls)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(5, 3)
    ax[0, 0].set_title('Position [m]')
    ax[0, 1].set_title('Velocity [m/s]')
    ax[0, 2].set_title('Acceleration [m/s$^2$]')
    ax[0, 0].set_ylabel('x')
    ax[1, 0].set_ylabel('y')
    ax[2, 0].set_ylabel('z')
    ax[3, 0].set_ylabel('mag')
    ax[4, 0].set_ylabel('error')
    for i in range(3):
        ax[i, 0].plot(states_sol[:, i])
        ax[i, 0].plot(mea_states[:, i], '--')
        ax[i, 1].plot(states_sol[:, i+3])
        ax[i, 1].plot(mea_states[:, i+3], '--')
        ax[i, 2].step(range(n_intervals), controls_sol[:, i], where='post')
        ax[i, 2].plot(accelerations[:, i], '--')
    r_sol = np.sum(states_sol[:, :3]**2, axis=1)**.5
    r_mea = np.sum(mea_states[:, :3]**2, axis=1)**.5
    ax[3, 0].plot(r_sol)
    ax[3, 0].plot(r_mea, '--')
    ax[4, 0].plot(r_sol-r_mea)

    v_sol = np.sum(states_sol[:, 3:6]**2, axis=1)**.5
    v_mea = np.sum(mea_states[:, 3:6]**2, axis=1)**.5
    ax[3, 1].plot(v_sol)
    ax[3, 1].plot(v_mea, '--')
    ax[4, 1].plot(v_sol-v_mea)

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