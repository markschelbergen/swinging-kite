import numpy as np
from scipy.optimize import least_squares
from attitude_check import plot_apparent_wind_velocity
from utils import add_panel_labels


def circle_residuals(c, x, y):
    return np.sqrt((x-c[0])**2 + (y-c[1])**2) - c[2]


r_limit = .25
center_estimate = [(0, 0.7, .1), (0, 0.4, .1)]
mark_points = [20, 40, 57, 76, 111, 146, 163, 177, 192]


def find_turn_specs(az, el):
    for j in range(2):
        res = least_squares(circle_residuals, center_estimate[j], args=(az, el), verbose=0)
        xc, yc, r = res.x
        if r < r_limit:
            return (xc, yc, r)
    return (np.nan, np.nan, np.nan)


def find_turns_for_rolling_window(flight_data):
    window = 9
    n_nans = (window-1)//2

    res = np.empty((flight_data.shape[0], 3))
    res[:] = np.nan
    for i in range(n_nans, flight_data.shape[0]-n_nans):
        if np.abs(flight_data.iloc[i].kite_azimuth) < .1:
            continue
        az = np.array(flight_data[i-n_nans:i+n_nans+1].kite_azimuth)
        el = np.array(flight_data[i-n_nans:i+n_nans+1].kite_elevation)
        res[i, :] = find_turn_specs(az, el)
    flight_data['azimuth_turn_center'] = res[:, 0]
    flight_data['elevation_turn_center'] = res[:, 1]
    flight_data['flag_turn'] = res[:, 2] < r_limit


def rigid_body_rotation_errors(omega, r_kite, v_kite, a_kite):
    v_om = np.cross(omega, r_kite)
    a_om = np.cross(omega, v_om)

    v_err = v_kite - v_om
    a_err = a_kite - a_om
    return np.hstack((v_err, a_err))


def determine_rigid_body_rotation(flight_data):
    omega_first_principle = np.empty((flight_data.shape[0], 3))
    omega_optimized = np.empty((flight_data.shape[0], 3))
    omega_magnitude = np.empty(flight_data.shape[0])

    v_kite_om = np.empty((flight_data.shape[0], 3))
    a_kite_om = np.empty((flight_data.shape[0], 3))
    v_kite_om_opt = np.empty((flight_data.shape[0], 3))
    a_kite_om_opt = np.empty((flight_data.shape[0], 3))

    for i, (idx, row) in enumerate(flight_data.iterrows()):
        # Project measured acceleration on sphere surface
        a_kite = np.array(list(row[['ax', 'ay', 'az']]))
        if i == flight_data.shape[0] - 1:
            a_kite = np.array(list(flight_data.loc[idx-1, ['ax', 'ay', 'az']]))
        v_kite = np.array(list(row[['vx', 'vy', 'vz']]))

        r_kite = np.array(list(row[['rx', 'ry', 'rz']]))

        if not np.isnan(row['elevation_turn_center']):  # Turning kite
            elt, azt = row['elevation_turn_center'], row['azimuth_turn_center']
            e_omega = np.array([np.cos(elt)*np.cos(azt), np.cos(elt)*np.sin(azt), np.sin(elt)])
            r_parallel = np.dot(r_kite, e_omega)*e_omega
            r_turn = r_kite - r_parallel

            e_v = np.cross(e_omega, r_kite)
            e_v = e_v/np.linalg.norm(e_v)
            v_kite_rot = np.dot(v_kite, e_v)*e_v

            om = np.cross(r_turn, v_kite_rot)/np.linalg.norm(r_turn)**2
            omega_magnitude[i] = np.linalg.norm(om)
            om_opt = least_squares(rigid_body_rotation_errors, om, args=(r_kite, v_kite, a_kite), verbose=0).x
        else:
            om = np.cross(r_kite, v_kite)/np.linalg.norm(r_kite)**2
            om_opt = least_squares(rigid_body_rotation_errors, om, args=(r_kite, v_kite, a_kite), verbose=0).x
            omega_magnitude[i] = np.linalg.norm(om)

        omega_first_principle[i, :] = om
        omega_optimized[i, :] = om_opt

        v_om_opt = np.cross(om_opt, r_kite)
        v_kite_om_opt[i] = v_om_opt
        a_om_opt = np.cross(om_opt, v_om_opt)
        a_kite_om_opt[i] = a_om_opt

        v_om = np.cross(om, r_kite)
        v_kite_om[i] = v_om
        a_om = np.cross(om, v_om)
        a_kite_om[i] = a_om

    flight_data['omx'] = omega_first_principle[:, 0]
    flight_data['omy'] = omega_first_principle[:, 1]
    flight_data['omz'] = omega_first_principle[:, 2]
    flight_data['omx_opt'] = omega_optimized[:, 0]
    flight_data['omy_opt'] = omega_optimized[:, 1]
    flight_data['omz_opt'] = omega_optimized[:, 2]
    # flight_data['om_straight'] = omega_straight
    # flight_data['om_turn'] = omega_turn
    flight_data['om'] = omega_magnitude

    flight_data['vx_om'] = v_kite_om[:, 0]
    flight_data['vy_om'] = v_kite_om[:, 1]
    flight_data['vz_om'] = v_kite_om[:, 2]
    flight_data['ax_om'] = a_kite_om[:, 0]
    flight_data['ay_om'] = a_kite_om[:, 1]
    flight_data['az_om'] = a_kite_om[:, 2]

    flight_data['vx_om_opt'] = v_kite_om_opt[:, 0]
    flight_data['vy_om_opt'] = v_kite_om_opt[:, 1]
    flight_data['vz_om_opt'] = v_kite_om_opt[:, 2]
    flight_data['ax_om_opt'] = a_kite_om_opt[:, 0]
    flight_data['ay_om_opt'] = a_kite_om_opt[:, 1]
    flight_data['az_om_opt'] = a_kite_om_opt[:, 2]


def plot_estimated_turn_center(flight_data, animate=False, vwx=10):
    import matplotlib.pyplot as plt
    ax = plt.figure(figsize=[5.8, 3]).gca()
    plt.subplots_adjust(top=1.0, bottom=0.16, left=0.1, right=.99)
    if animate:
        for i in range(flight_data.shape[0]):
            ax.cla()
            ax.set_xlim([-20, 20])
            ax.set_ylim([24, 46])
            ax.set_xlabel('Azimuth [$^\circ$]')
            ax.set_ylabel('Elevation [$^\circ$]')
            ax.set_aspect('equal')

            ax.plot(flight_data.iloc[:i]['kite_azimuth']*180./np.pi, flight_data.iloc[:i]['kite_elevation']*180./np.pi)
            ax.plot(flight_data.iloc[:i]['azimuth_turn_center']*180./np.pi, flight_data.iloc[:i]['elevation_turn_center']*180./np.pi, linewidth=.5, color='grey')
            ax.plot(flight_data.iloc[i]['azimuth_turn_center']*180./np.pi, flight_data.iloc[i]['elevation_turn_center']*180./np.pi, 's')
            plt.pause(0.001)
    else:
        ax.set_xlim([-23, 23])
        ax.set_ylim([24, 47])
        ax.set_xlabel('Azimuth [$^\circ$]')
        ax.set_ylabel('Elevation [$^\circ$]')
        ax.set_aspect('equal')

        az, el = flight_data['kite_azimuth']*180./np.pi, flight_data['kite_elevation']*180./np.pi
        ax.plot(np.where(~flight_data['flag_turn'], az, np.nan), np.where(~flight_data['flag_turn'], el, np.nan), color='C0', label='Straight')
        ax.plot(np.where(flight_data['flag_turn'], az, np.nan), np.where(flight_data['flag_turn'], el, np.nan), '--', color='C0', label='Turn')

        ax.plot(flight_data['azimuth_turn_center']*180./np.pi, flight_data['elevation_turn_center']*180./np.pi, linewidth=.8, color='grey', label='Turn center')

        for j, im in enumerate(mark_points):
            row = flight_data.iloc[im]

            if j == 0:
                lbl = 'Heading'
            else:
                lbl = None
            plot_apparent_wind_velocity(ax, row, vwx, .3, color='g', linestyle=':', label=lbl)

            az_tc, el_tc = row['azimuth_turn_center']*180./np.pi, row['elevation_turn_center']*180./np.pi
            if np.isnan(az_tc):
                marker = 's'
            else:
                marker = 'o'
            ax.plot(az_tc, el_tc, marker, mfc="white", alpha=1, ms=6, mec='C{}'.format(j))

            az, el = row['kite_azimuth']*180./np.pi, row['kite_elevation']*180./np.pi
            ax.plot(az, el, marker, mfc="white", alpha=1, ms=12, mec='C{}'.format(j))
            ax.plot(az, el, marker='${}$'.format(j+1), alpha=1, ms=7, mec='C{}'.format(j))
        ax.legend(loc=9)
        ax.grid()


def visualize_estimated_rotation_vector(flight_data, animate=False):
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    from utils import plot_vector

    def calc_sphere_coords(x, y, z):
        r = (x**2 + y**2 + z**2)**.5
        el = np.arcsin(z/r)*180./np.pi
        az = np.arctan2(y, x)*180./np.pi
        return az, el, r

    om_opt_sphere = calc_sphere_coords(flight_data.omx_opt, flight_data.omy_opt, flight_data.omz_opt)
    om_sphere = calc_sphere_coords(flight_data.omx, flight_data.omy, flight_data.omz)

    # # Plot all cartesian components of omega estimated on first princples and data-inferred
    # fig, ax = plt.subplots(3, 2)
    # ax[0, 0].get_shared_y_axes().join(*ax[:, 0])
    # ax[0, 0].plot(flight_data.time, flight_data.omx_opt, color='C1', label='opt')
    # ax[0, 0].plot(flight_data.time, flight_data.omx, color='C2', label='est')
    # ax[0, 0].set_ylabel('x [rad/s]')
    # ax[1, 0].plot(flight_data.time, flight_data.omy_opt, color='C1', label='opt')
    # ax[1, 0].plot(flight_data.time, flight_data.omy, color='C2', label='est')
    # ax[1, 0].set_ylabel('y [rad/s]')
    # ax[2, 0].plot(flight_data.time, flight_data.omz_opt, color='C1', label='opt')
    # ax[2, 0].plot(flight_data.time, flight_data.omz, color='C2', label='est')
    # ax[2, 0].set_ylabel('z [rad/s]')
    #
    # # Plot all spherical components of omega estimated on first princples and data-inferred
    # ax[0, 1].plot(flight_data.time, om_opt_sphere[0], color='C1', label='opt')
    # ax[0, 1].plot(flight_data.time, om_sphere[0], color='C2', label='est')
    # ax[0, 1].set_ylabel('Azimuth [deg]')
    # ax[1, 1].plot(flight_data.time, om_opt_sphere[1], color='C1', label='opt')
    # ax[1, 1].plot(flight_data.time, om_sphere[1], color='C2', label='est')
    # ax[1, 1].set_ylabel('Elevation [deg]')
    # ax[2, 1].plot(flight_data.time, om_opt_sphere[2], color='C1', label='opt')
    # ax[2, 1].plot(flight_data.time, om_sphere[2], color='C2', label='est')
    # ax[2, 1].set_ylabel('Magnitude [rad/s]')
    # ax[2, 1].legend()

    # Plot all cartesian components of omega estimated on first princples and data-inferred
    fig, ax = plt.subplots(4, 2)
    ax[0, 0].set_title('Velocity [m/s]')
    ax[0, 0].plot(flight_data.time, flight_data.vx_om, color='C1', label='om')
    ax[0, 0].plot(flight_data.time, flight_data.vx_om_opt, color='C3', label='om opt')
    ax[0, 0].plot(flight_data.time, flight_data.vx, color='C2', label='data')
    ax[0, 0].plot(flight_data.time, flight_data.kite_0_vx, ':', label='mea')
    ax[0, 0].set_ylabel('x')
    ax[1, 0].plot(flight_data.time, flight_data.vy_om, color='C1', label='om')
    ax[1, 0].plot(flight_data.time, flight_data.vy_om_opt, color='C3', label='om opt')
    ax[1, 0].plot(flight_data.time, flight_data.vy, color='C2', label='data')
    ax[1, 0].plot(flight_data.time, flight_data.kite_0_vy, ':', label='mea')
    ax[1, 0].set_ylabel('y')
    ax[2, 0].plot(flight_data.time, flight_data.vz_om, color='C1', label='om')
    ax[2, 0].plot(flight_data.time, flight_data.vz_om_opt, color='C3', label='omopt')
    ax[2, 0].plot(flight_data.time, flight_data.vz, color='C2', label='data')
    ax[2, 0].plot(flight_data.time, flight_data.kite_0_vz, ':', label='mea')
    ax[2, 0].set_ylabel('z')
    ax[3, 0].plot(flight_data.time, (flight_data.vx_om**2+flight_data.vy_om**2+flight_data.vz_om**2)**.5, color='C1', label='om')
    ax[3, 0].plot(flight_data.time, (flight_data.vx_om_opt**2+flight_data.vy_om_opt**2+flight_data.vz_om_opt**2)**.5, color='C3', label='om opt')
    ax[3, 0].plot(flight_data.time, (flight_data.vx**2+flight_data.vy**2+flight_data.vz**2)**.5, color='C2', label='data')
    ax[3, 0].plot(flight_data.time, (flight_data.kite_0_vx**2+flight_data.kite_0_vy**2+flight_data.kite_0_vz**2)**.5, ':', label='mea')
    ax[3, 0].set_ylabel('mag')

    ax[0, 1].set_title('Acceleration [m/s^2]')
    ax[0, 1].plot(flight_data.time, flight_data.ax_om, color='C1', label='om')
    ax[0, 1].plot(flight_data.time, flight_data.ax_om_opt, color='C3', label='om opt')
    ax[0, 1].plot(flight_data.time, flight_data.ax, color='C2', label='data')
    ax[0, 1].plot(flight_data.time, flight_data.kite_1_ax, label='mea')
    ax[1, 1].plot(flight_data.time, flight_data.ay_om, color='C1', label='om')
    ax[1, 1].plot(flight_data.time, flight_data.ay_om_opt, color='C3', label='om opt')
    ax[1, 1].plot(flight_data.time, flight_data.ay, color='C2', label='data')
    ax[1, 1].plot(flight_data.time, flight_data.kite_1_ay, label='mea')
    ax[2, 1].plot(flight_data.time, flight_data.az_om, color='C1', label='om')
    ax[2, 1].plot(flight_data.time, flight_data.az_om_opt, color='C3', label='om opt')
    ax[2, 1].plot(flight_data.time, flight_data.az, color='C2', label='data')
    ax[2, 1].plot(flight_data.time, flight_data.kite_1_az, label='mea')
    ax[3, 1].plot(flight_data.time, (flight_data.ax**2+flight_data.ay**2+flight_data.az**2)**.5, color='C1', label='reconstructed', linewidth=2.5)
    ax[3, 1].plot(flight_data.time, (flight_data.kite_1_ax**2+flight_data.kite_1_ay**2+flight_data.kite_1_az**2)**.5, ':', color='C2', label='flight data')
    ax[3, 1].plot(flight_data.time, (flight_data.ax_om**2+flight_data.ay_om**2+flight_data.az_om**2)**.5, color='C3', label=r'$\omega_{\rm rb-gc/turn}$', linewidth=1)
    ax[3, 1].plot(flight_data.time, (flight_data.ax_om_opt**2+flight_data.ay_om_opt**2+flight_data.az_om_opt**2)**.5, '--', color='C0', label=r'$\omega_{\rm rb-opt}$')
    ax[3, 1].legend()

    # Plot only the magnitude for the paper
    fig, ax = plt.subplots(3, 1, figsize=[5.8, 4.8], sharex=True)
    plt.subplots_adjust(top=0.985, bottom=0.105, left=0.185, right=0.985, hspace=0.125)
    ax[0].plot(flight_data.time, flight_data.om, color='C3', label=r'$\omega_{\rm rb-gc/turn}$', linewidth=1)
    ax[0].plot(flight_data.time, om_opt_sphere[2], '--', color='C0', label=r'$\omega_{\rm rb-opt}$')
    ax[0].set_xlim([flight_data['time'].iloc[0], flight_data['time'].iloc[-1]])
    ax[0].set_ylim([0, None])
    ax[0].set_ylabel('Rotational\nspeed [rad s$^{-1}$]')
    ax[0].legend(ncol=2)

    ax[1].plot(flight_data.time, (flight_data.vx**2+flight_data.vy**2+flight_data.vz**2)**.5, color='C1', label='reconstructed', linewidth=2.5)
    ax[1].plot(flight_data.time, (flight_data.kite_0_vx**2+flight_data.kite_0_vy**2+flight_data.kite_0_vz**2)**.5, ':', color='C2', label='flight data')
    ax[1].plot(flight_data.time, (flight_data.vx_om**2+flight_data.vy_om**2+flight_data.vz_om**2)**.5, color='C3', linewidth=1)  #, label=r'$\omega_{\rm rb-gc/turn}$')
    ax[1].plot(flight_data.time, (flight_data.vx_om_opt**2+flight_data.vy_om_opt**2+flight_data.vz_om_opt**2)**.5, '--', color='C0')  #, label=r'$\omega_{\rm rb-opt}$')
    ax[1].set_ylabel('Canopy speed\n[m s$^{-1}$]')
    ax[1].legend(ncol=2)

    ax[2].plot(flight_data.time, (flight_data.ax**2+flight_data.ay**2+flight_data.az**2)**.5, color='C1', label='reconstructed', linewidth=2.5)
    ax[2].plot(flight_data.time, (flight_data.kite_1_ax**2+flight_data.kite_1_ay**2+flight_data.kite_1_az**2)**.5, ':', color='C2', label='flight data')
    ax[2].plot(flight_data.time, (flight_data.ax_om**2+flight_data.ay_om**2+flight_data.az_om**2)**.5, color='C3', linewidth=1)
    ax[2].plot(flight_data.time, (flight_data.ax_om_opt**2+flight_data.ay_om_opt**2+flight_data.az_om_opt**2)**.5, '--', color='C0')
    ax[2].set_ylabel('Canopy\nacceleration [m s$^{-2}$]')

    add_panel_labels(ax, offset_x=.22)
    ax[2].set_xlabel('Time [s]')
    for a in ax:
        a.grid()
        y1 = a.get_ylim()[1]
        a.set_ylim([0, y1])
        a.fill_between(flight_data.time, 0, y1, where=flight_data['flag_turn'], facecolor='lightgrey', alpha=0.5)

    # mark_points = np.array([44, 105, 151, 210]) - 44  #, 250, 320]
    #
    # # Plot how the orientation (elevation+azimuth) of the rotational vector changes
    # def plot_omega_evolution(i, ti=None):
    #     idx2i = flight_data.index[:i+1]
    #     ax[0].plot(flight_data.loc[idx2i, 'kite_azimuth']*180./np.pi, flight_data.loc[idx2i, 'kite_elevation']*180./np.pi)
    #     if ti is not None:
    #         ax[0].text(10, 39, "{:.2f} s".format(ti))
    #     ax[0].set_xlim([-20, 20])
    #     ax[0].set_ylim([28, 42])
    #     ax[0].set_xlabel('Kite position azimuth [deg]')
    #     ax[0].set_ylabel('Kite position elevation [deg]')
    #     # ax[1].plot(om_opt_sphere[0].iloc[:i+1], om_opt_sphere[1].iloc[:i+1], color='C1', label='opt', linewidth=.5)
    #     # ax[1].plot(om_opt_sphere[0].iloc[i], om_opt_sphere[1].iloc[i], 's', color='C1')
    #     ax[1].plot(om_sphere[0].iloc[:i+1], om_sphere[1].iloc[:i+1], color='C2', label='est', linewidth=.5)
    #     ax[1].plot(om_sphere[0].iloc[i], om_sphere[1].iloc[i], 's', color='C2')
    #     ax[1].set_xlim([-180, 180])
    #     ax[1].set_ylim([-80, 80])
    #     ax[1].legend()
    #     ax[1].set_xlabel('Rotational vector azimuth [deg]')
    #     ax[1].set_ylabel('Rotational vector elevation [deg]')
    #
    #     for j, im in enumerate(mark_points):
    #         if im > i:
    #             break
    #         az, el = flight_data.loc[idx2i[im], 'kite_azimuth']*180./np.pi, flight_data.loc[idx2i[im], 'kite_elevation']*180./np.pi
    #         ax[0].plot(az, el, 'o', mfc="white", alpha=1, ms=12, mec='C{}'.format(j))
    #         ax[0].plot(az, el, marker='${}$'.format(j+1), alpha=1, ms=7, mec='C{}'.format(j))
    #         ax[1].plot(om_sphere[0].iloc[im], om_sphere[1].iloc[im], 'o', mfc="white", alpha=1, ms=12, mec='C{}'.format(j))
    #         ax[1].plot(om_sphere[0].iloc[im], om_sphere[1].iloc[im], marker='${}$'.format(j+1), alpha=1, ms=7, mec='C{}'.format(j))
    #
    # fig, ax = plt.subplots(1, 2)
    # if animate:
    #     for i, ti in enumerate(flight_data.time):
    #         for a in ax: a.cla()
    #         plot_omega_evolution(i, ti)
    #         plt.pause(0.001)
    # else:
    #     plot_omega_evolution(flight_data.shape[0]-1)
    #
    # # Plot rotational and position vector in 3d - also depicts the magnitude of the rotational vector.
    # plt.figure(figsize=(8, 6))
    # ax3d = plt.axes(projection='3d')
    # ax3d.plot3D(flight_data['rx'], flight_data['ry'], flight_data['rz'], label='')
    # ax3d.set_xlim([0, 250])
    # ax3d.set_ylim([-125, 125])
    # ax3d.set_zlim([0, 250])
    # ax3d.set_xlabel("x [m]")
    # ax3d.set_ylabel("y [m]")
    # ax3d.set_zlabel("z [m]")
    # for j, im in enumerate(mark_points):
    #     idx = flight_data.index[im]
    #     om = flight_data.loc[idx, ['omx', 'omy', 'omz']]
    #     plot_vector(np.zeros(3), om, ax3d, 1e2, color='C{}'.format(j), label='{}'.format(j+1))
    #     plot_vector(np.zeros(3), om/np.linalg.norm(om), ax3d, 3e2, color='C{}'.format(j), linestyle=':')
    #     plot_vector(np.zeros(3), flight_data.loc[idx, ['rx', 'ry', 'rz']], ax3d, 1, color='C{}'.format(j), linestyle='--')
    # ax3d.legend()


if __name__ == "__main__":
    from utils import read_and_transform_flight_data
    from matplotlib.pyplot import show
    flight_data = read_and_transform_flight_data(make_kinematics_consistent=True)
    print("Min. and max. height: {:.2f} and {:.2f}".format(flight_data['rz'].min(), flight_data['rz'].max()))
    print("Start and height radial position: {:.2f} and {:.2f}".format(flight_data.iloc[0]['kite_distance'], flight_data.iloc[-1]['kite_distance']))
    # import matplotlib.pyplot as plt
    # plt.plot(flight_data.time, flight_data.kite_azimuth*180./np.pi)
    # plt.grid()
    # plt.show()
    # exit()
    determine_rigid_body_rotation(flight_data)
    plot_estimated_turn_center(flight_data)
    visualize_estimated_rotation_vector(flight_data)
    show()