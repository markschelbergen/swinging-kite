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
    opti.set_initial(states, df[['rx', 'ry', 'rz', 'vx', 'vy', 'vz']].values)

    accelerations = np.vstack([[np.gradient(df['vx'])/.1], [np.gradient(df['vy'])/.1], [np.gradient(df['vz'])/.1]]).T
    opti.set_initial(controls, accelerations[:-1, :])

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