import numpy as np
import matplotlib.pyplot as plt
import pickle
from utils import read_and_transform_flight_data, add_panel_labels, plot_flight_sections
from turning_center import mark_points


def plot_offaxial_tether_displacement(pos_tau, ax=None, ls='-', plot_rows=[0, 1], plot_instances=mark_points[:5]):
    if ax is None:
        set_legend = True

        fig, ax = plt.subplots(2, 2, sharey=True, figsize=(6.4, 5.2))
        ax[0, 0].set_ylim([0, 300])
        ax[0, 1].invert_xaxis()
        ax[1, 1].invert_xaxis()
        wspace = .200
        plt.subplots_adjust(top=0.885, bottom=0.105, left=0.15, right=0.99, hspace=0.07, wspace=wspace)
        ax[0, 0].set_ylabel("Radial position [m]")
        ax[1, 0].set_ylabel("Radial position [m]")
        ax[1, 0].set_xlabel("Up-apparent-wind position [m]")
        ax[1, 1].set_xlabel("Cross-apparent-wind position [m]")
        ax[1, 0].get_shared_x_axes().join(*ax[:, 0])
        ax[1, 1].get_shared_x_axes().join(*ax[:, 1])
        for a in ax.reshape(-1): a.grid()
        ax[0, 0].tick_params(labelbottom=False)
        ax[0, 1].tick_params(labelbottom=False)
        ax[1, 0].set_xlim([-6, 1])
        ax[1, 1].set_xlim([4.5, -2.5])
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


def combine_results_of_different_analyses():
    flight_data = read_and_transform_flight_data()

    fig, ax_ypr = plt.subplots(2, 1, sharex=True, figsize=[6.4, 4.5])
    plt.subplots_adjust(top=0.78, bottom=0.135, left=0.16, right=0.99,)
    # plt.suptitle("3-2-1 Euler angles between tangential\nand bridle ref. frames")
    ax_ypr[0].set_ylabel("Pitch [$^\circ$]")
    ax_ypr[1].set_ylabel("Roll [$^\circ$]")
    ax_ypr[1].set_xlabel("Instance label [-]")
    ax2 = ax_ypr[0].secondary_xaxis("top")
    ax2.set_xlabel("Time [s]")

    n_tether_elements = [30, 1]
    linestyles = ['-', '--']
    linewidths = [2.5, 1.5]
    ax_tether_shape = None
    for i, (n_te, ls, lw) in enumerate(zip(n_tether_elements, linestyles, linewidths)):
        with open("results/time_invariant_results{}.pickle".format(n_te), 'rb') as f:
            res = pickle.load(f)
        if i == 1:
            plot_rows = [0]
        else:
            plot_rows = [0, 1]
        ax_tether_shape = plot_offaxial_tether_displacement(res['offaxial_tether_shape'], ax_tether_shape, ls=ls,
                                                            plot_rows=plot_rows)

        ax_ypr[0].plot(flight_data.time, res['pitch_bridle']*180./np.pi, ls=ls, linewidth=lw, label=r'Steady-rotation N='+str(n_te))
        ax_ypr[1].plot(flight_data.time, res['roll_bridle']*180./np.pi, ls=ls, linewidth=lw, label=r'Steady-rotation N='+str(n_te))

    with open("results/dynamic_results30.pickle", 'rb') as f:
        res = pickle.load(f)
    ax_tether_shape = plot_offaxial_tether_displacement(res['offaxial_tether_shape'], ax_tether_shape, ls='-.', plot_rows=[1])
    ax_ypr[0].plot(flight_data.time, res['pitch_bridle']*180./np.pi, label=r'Dynamic N=30')
    ax_ypr[1].plot(flight_data.time, res['roll_bridle']*180./np.pi, label=r'Dynamic N=30')

    ax_ypr[0].plot(flight_data.time, flight_data.pitch0_tau * 180. / np.pi, '-.', label='Sensor 0')
    ax_ypr[0].plot(flight_data.time, flight_data.pitch1_tau * 180. / np.pi, '-.', label='Sensor 1')
    ax_ypr[1].plot(flight_data.time, flight_data.roll0_tau * 180. / np.pi, '-.', label='Sensor 0')
    ax_ypr[1].plot(flight_data.time, flight_data.roll1_tau * 180. / np.pi, '-.', label='Sensor 1')

    ax_ypr[0].legend(bbox_to_anchor=(.05, 1.4, .9, .5), loc="lower left", mode="expand",
                   borderaxespad=0, ncol=3)
    for a in ax_ypr: a.set_xticks(flight_data['time'].iloc[mark_points])
    ax_ypr[1].set_xticklabels([str(i+1) for i in range(len(mark_points))])
    for a in ax_ypr: a.grid()
    for a in ax_ypr: plot_flight_sections(a, flight_data)
    add_panel_labels(ax_ypr, .18)
    ax_ypr[0].set_xlim([flight_data.iloc[0]['time'], flight_data.iloc[-1]['time']])

    add_panel_labels(ax_tether_shape, (.38, .15))


if __name__ == "__main__":
    combine_results_of_different_analyses()  # Plots figures 8 and 9
    plt.show()