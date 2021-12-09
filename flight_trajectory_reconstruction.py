# Borrowed from tether-model project
import casadi as ca
import numpy as np


def setup_integrator_kinematic_model(tf, solver='idas'):
    # ODE for kinematic model
    r = ca.SX.sym('r', 4)
    v = ca.SX.sym('v', 4)
    x = ca.vertcat(r, v)
    a = ca.SX.sym('a', 4)
    dx = ca.vertcat(v, a)

    # Create an integrator
    ode = {'x': x, 'ode': dx, 'p': a}

    intg = ca.integrator('intg', solver, ode, {'tf': tf})
    return intg


def find_acceleration_matching_kite_trajectory(df, solver='idas'):
    n_intervals = df.shape[0]-1
    tf = .1

    intg = setup_integrator_kinematic_model(tf, solver)

    opti = ca.casadi.Opti()

    # Decision variables for states
    states = opti.variable(n_intervals+1, 8)
    # Decision variables for control vector
    controls = opti.variable(n_intervals, 4)

    # Gap-closing shooting constraints
    for k in range(n_intervals):
        res = intg(x0=states[k, :], p=controls[k, :])
        opti.subject_to(states[k+1, :].T == res["xf"])

    # Initial guesses
    mea_pos = df[['rx', 'ry', 'rz']].values
    opti.set_initial(states[:, :3], mea_pos)
    opti.set_initial(states[:, 3], df.kite_distance.values)
    mea_speed = df[['vx', 'vy', 'vz']].values
    opti.set_initial(states[:, 4:7], mea_speed)
    opti.set_initial(states[:, 7], df.ground_tether_reelout_speed.values)

    accelerations = df[['kite_1_ax', 'kite_1_ay', 'kite_1_az']].values
    opti.set_initial(controls[:, :3], accelerations[:-1, :3])

    opti.subject_to(ca.sum2(states[:, :3]**2)**.5 == states[:, 3])
    obj = ca.sumsqr(states[:, :3] - mea_pos)

    weight_radial_speed = 5
    obj += ca.sumsqr(weight_radial_speed*(states[:, 7] - df.ground_tether_reelout_speed.values))

    weight_control_steps = 1/25
    control_steps = controls[1:, :3]-controls[:-1, :3]
    obj += ca.sumsqr(weight_control_steps*control_steps)
    opti.minimize(obj)

    opti.subject_to(opti.bounded(-3, control_steps, 3))

    # solve optimization problem
    opti.solver('ipopt', {'ipopt':  {'max_iter': 50}})

    sol = opti.solve()
    get_values = sol.value

    states_sol = get_values(states)
    controls_sol = get_values(controls)

    return states_sol, controls_sol


def find_acceleration_matching_kite_trajectory2(df, verify=False, solver='idas', weights=None, r0=None):
    n_intervals = df.shape[0]-1
    tf = .1

    intg = setup_integrator_kinematic_model(tf, solver)

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

    if weights is None:
        weights = np.ones(mea_states.shape)
        # weights[:, 3:6] = .7
        # weights[df.flag_turn, :3] = .1
        # weights[~df.flag_turn, :3] = 2

    if r0 is not None:
        opti.subject_to(states[0, :3] == np.array([r0]))

    # opti.subject_to(opti.bounded(-3, controls[1:, :]-controls[:-1, :], 3))
    obj = ca.sumsqr(weights*(states - mea_states))
    # obj = obj + ca.sumsqr(ca.sum2(states[:, 3:]**2)**.5 - ca.sum2(mea_states[:, 3:]**2)**.5)
    opti.minimize(obj)

    # solve optimization problem
    opti.solver('ipopt')

    sol = opti.solve()

    states_sol = sol.value(states)
    controls_sol = sol.value(controls)

    return states_sol, controls_sol