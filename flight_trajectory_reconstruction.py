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


def apply_low_pass_filter(data, cutoff_freq=.3, fillna=True):
    from scipy.signal import butter, filtfilt
    fs = 10.0       # sample rate, Hz
    nyq = 0.5 * fs  # Nyquist Frequency

    order = 2  # Trend can be well captured with order of 2
    normal_cutoff = cutoff_freq / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    if fillna:
        data = data.interpolate()
        if np.isnan(data.iloc[0]):
            data.iloc[0] = data.iloc[1]
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def plot_kinematics():
    import matplotlib.pyplot as plt
    from utils import read_and_transform_flight_data, add_panel_labels, plot_flight_sections2

    flight_data_raw = read_and_transform_flight_data(False, 65)  # Read flight data.
    flight_data_rec = read_and_transform_flight_data(True, 65)  # Read flight data.

    r = np.sum(flight_data_raw[['rx', 'ry', 'rz']].values**2, axis=1)**.5
    r_c = np.sum(flight_data_rec[['rx', 'ry', 'rz']].values**2, axis=1)**.5

    radial_speed = np.sum(flight_data_raw[['rx', 'ry', 'rz']].values*flight_data_raw[['vx', 'vy', 'vz']].values, axis=1)/r
    radial_speed_c = np.sum(flight_data_rec[['rx', 'ry', 'rz']].values*flight_data_rec[['vx', 'vy', 'vz']].values, axis=1)/r_c

    fig, ax = plt.subplots(4, 2, sharex=True, figsize=(7.2, 6))
    plt.subplots_adjust(left=.13, bottom=.09, right=.99, top=.94, wspace=.26, hspace=.15)
    ax[0, 0].set_title('Position [m]')
    ax[0, 1].set_title('Velocity [m/s]')
    ax[0, 0].set_ylabel('x')
    ax[1, 0].set_ylabel('y')
    ax[2, 0].set_ylabel('z')
    ax[3, 0].set_ylabel('r')
    ax[-1, 0].set_xlabel('Time [s]')
    ax[-1, 1].set_xlabel('Time [s]')

    ax[0, 0].plot(flight_data_raw.time, flight_data_raw.rx, label='flight data')
    ax[0, 0].plot(flight_data_rec.time, flight_data_rec.rx, '--', label='reconstructed')
    ax[0, 0].legend()
    ax[1, 0].plot(flight_data_raw.time, flight_data_raw.ry)
    ax[1, 0].plot(flight_data_rec.time, flight_data_rec.ry, '--')
    ax[2, 0].plot(flight_data_raw.time, flight_data_raw.rz)
    ax[2, 0].plot(flight_data_rec.time, flight_data_rec.rz, '--')

    ax[0, 1].plot(flight_data_raw.time, flight_data_raw.vx)
    # ax[0, 1].plot(flight_data_raw.time[:-1], np.diff(flight_data_raw.rx)/.1, '--')
    ax[0, 1].plot(flight_data_rec.time, flight_data_rec.vx, '--')
    ax[1, 1].plot(flight_data_raw.time, flight_data_raw.vy)
    # ax[1, 1].plot(flight_data_raw.time[:-1], np.diff(flight_data_raw.ry)/.1, '--')
    ax[1, 1].plot(flight_data_rec.time, flight_data_rec.vy, '--')
    ax[2, 1].plot(flight_data_raw.time, flight_data_raw.vz)
    # ax[2, 1].plot(flight_data_raw.time[:-1], np.diff(flight_data_raw.rz)/.1, '--')
    ax[2, 1].plot(flight_data_rec.time, flight_data_rec.vz, '--')

    ax[3, 0].plot(flight_data_raw.time, r)
    ax[3, 0].plot(flight_data_rec.time, r, '--')
    # ax[3, 0].plot(flight_data_raw.time, flight_data_raw.rr, '-.')

    drdt = apply_low_pass_filter(np.diff(r), cutoff_freq=1, fillna=False)/.1
    ax[3, 1].plot(flight_data_raw.time[:-1], drdt, label='$\hat{\dot{r}}$')
    ax[3, 1].plot(flight_data_raw.time, radial_speed, color='r', label=r'$\hat{v}_\mathrm{r}$', linewidth=.7)
    ax[3, 1].plot(flight_data_raw.time, radial_speed_c, '--', color='C1')
    ax[3, 1].plot(flight_data_raw.time, flight_data_raw.ground_tether_reelout_speed, 'k:', label='$\hat{\dot{l}}_\mathrm{t}$')
    ax[3, 1].legend(ncol=3)

    plot_flight_sections2(ax, flight_data_raw, False)
    ax[0, 0].set_xlim([flight_data_raw['time'].iloc[0], flight_data_raw['time'].iloc[-1]])
    add_panel_labels(ax, [.33, .22])

    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(6.4, 4.8))
    ax[0].plot(flight_data_raw.time, flight_data_raw.kite_1_ax, linewidth=.5)
    ax[0].plot(flight_data_raw.time, flight_data_raw.ax)
    ax[1].plot(flight_data_raw.time, flight_data_raw.kite_1_ay, linewidth=.5)
    ax[1].plot(flight_data_raw.time, flight_data_raw.ay)
    ax[2].plot(flight_data_raw.time, flight_data_raw.kite_1_az, linewidth=.5)
    ax[2].plot(flight_data_raw.time, flight_data_raw.az)
    ax[3].plot(flight_data_raw.time, flight_data_raw.ddl)

    plot_flight_sections2(ax, flight_data_raw, False)
    plt.show()


def plot_reconstruction():
    from scipy.integrate import cumtrapz
    import matplotlib.pyplot as plt
    from utils import read_and_transform_flight_data, add_panel_labels, plot_flight_sections2

    flight_data_raw = read_and_transform_flight_data(False, 65)  # Read flight data.
    flight_data_rec = read_and_transform_flight_data(True, 65)  # Read flight data.

    r = np.sum(flight_data_raw[['rx', 'ry', 'rz']].values**2, axis=1)**.5
    r_c = np.sum(flight_data_rec[['rx', 'ry', 'rz']].values**2, axis=1)**.5

    radial_speed = np.sum(flight_data_raw[['rx', 'ry', 'rz']].values*flight_data_raw[['vx', 'vy', 'vz']].values, axis=1)/r
    radial_speed_c = np.sum(flight_data_rec[['rx', 'ry', 'rz']].values*flight_data_rec[['vx', 'vy', 'vz']].values, axis=1)/r_c

    fig, ax = plt.subplots(4, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1, 2, 1]}, figsize=[6.4, 6.4])
    plt.subplots_adjust(left=.17, bottom=.08, right=.98, top=.98, hspace=.1)
    ax[0].set_ylabel('Length / distance [m]')
    lt = cumtrapz(flight_data_raw.ground_tether_reelout_speed.values, initial=0)*.1
    ax[0].plot(flight_data_raw.time, lt, ':', color='k', label='$\int \hat{\dot{l}}_\mathrm{t}$')
    ax[0].plot(flight_data_raw.time, r - r[0], label='$\hat{r}_\mathrm{c} - \hat{r}_\mathrm{c,0}$')
    ax[0].plot(flight_data_raw.time, r_c - r_c[0], '--', label='$r_\mathrm{c} - r_\mathrm{c,0}$')
    ax[0].legend()
    ax[1].set_ylabel('Residual\nlength [m]')
    ax[1].plot(flight_data_raw.time, lt-(r - r[0]), label='$\Delta \hat{l}_\mathrm{t}$')
    ax[1].plot(flight_data_raw.time, lt-(r_c - r_c[0]), label='$\Delta l_\mathrm{t}$')
    ax[1].legend(ncol=2)

    drdt = apply_low_pass_filter(np.diff(r), cutoff_freq=1, fillna=False)/.1
    ax[2].plot(flight_data_raw.time[:-1], drdt, label='$\hat{\dot{r}}_\mathrm{c}$')
    ax[2].plot(flight_data_raw.time, radial_speed, color='r', label=r'$\hat{v}_\mathrm{c,r}$', linewidth=.7)
    ax[2].plot(flight_data_raw.time, radial_speed_c, '-', color='C1', label=r'$v_\mathrm{c,r}$')
    ax[2].plot(flight_data_raw.time, flight_data_raw.ground_tether_reelout_speed, 'k:', label='$\hat{\dot{l}}_\mathrm{t}$')
    ax[2].legend(ncol=4)
    ax[2].set_ylabel('Speed [m/s]')

    ax[3].plot(flight_data_raw.time, radial_speed_c-flight_data_raw.ground_tether_reelout_speed, label='$v_\mathrm{c,r} - \hat{\dot{l}}_\mathrm{t}$')
    ax[3].legend()
    ax[3].set_ylabel('Residual speed [m/s]')

    plot_flight_sections2(ax, flight_data_raw, False)
    add_panel_labels(ax, offset_x=.2)
    ax[-1].set_xlabel('Time [s]')
    ax[-1].set_xlim([flight_data_raw['time'].iloc[0], flight_data_raw['time'].iloc[-1]])

    plt.show()


if __name__ == "__main__":
    # plot_kinematics()
    plot_reconstruction()