import matplotlib.pyplot as plt
import numpy as np
from utils import plot_flight_sections2, read_and_transform_flight_data, add_panel_labels
from scipy.stats import linregress


def plot_relations():
    flight_data = read_and_transform_flight_data(i_cycle=65)
    x = np.array([flight_data.kite_actual_steering.min(), flight_data.kite_actual_steering.max()])

    fig, ax = plt.subplots(4, 2, figsize=[7.2, 6.4], gridspec_kw={'width_ratios': [3, 1]})
    plt.subplots_adjust(top=0.985, bottom=0.085, left=0.15, right=0.99, hspace=0.1, wspace=0.225)
    ax[0, 0].set_ylabel("Steering input [%]")
    ax[1, 0].set_ylabel("Twist between\nstruts [$^\circ$]")
    ax[2, 0].set_ylabel("Roll [$^\circ$]")
    ax[3, 0].set_ylabel("Yaw rate [$^\circ$ s$^{-1}$]")
    ax[3, 0].set_xlabel("Time [s]")
    ax[3, 1].set_xlabel("Steering input [%]")

    ax[0, 0].plot(flight_data.time, flight_data.kite_actual_steering, label='u')
    ax[0, 1].axis('off')

    ax[1, 0].plot(flight_data.time, (flight_data.pitch0_tau-flight_data.pitch1_tau)*180./np.pi)
    ax[1, 1].scatter(flight_data.kite_actual_steering, (flight_data.pitch0_tau-flight_data.pitch1_tau)*180./np.pi, s=.1)
    res = linregress(flight_data.kite_actual_steering, flight_data.pitch0_tau-flight_data.pitch1_tau)
    ax[1, 1].plot(x, res.slope*x*180./np.pi, '--', color='C1')  #(res.intercept + res.slope*x)
    ax[1, 1].text(.55, .7, "r={:.2f}".format(res.rvalue), transform=ax[1, 1].transAxes)
    ax[1, 0].plot(flight_data.time, flight_data.kite_actual_steering*res.slope*180./np.pi, '--', color='C1')

    ax[2, 0].plot(flight_data.time, flight_data.roll0_tau*180./np.pi)
    ax[2, 1].scatter(flight_data.kite_actual_steering, flight_data.roll0_tau*180./np.pi, s=.1)
    res = linregress(flight_data.kite_actual_steering, flight_data.roll0_tau)
    ax[2, 1].plot(x, res.slope*x*180./np.pi, '--', color='C1')
    ax[2, 1].text(.55, .7, "r={:.2f}".format(res.rvalue), transform=ax[2, 1].transAxes)
    ax[2, 0].plot(flight_data.time, flight_data.kite_actual_steering*res.slope*180./np.pi, '--', color='C1')

    ax[3, 0].plot(flight_data.time, flight_data.kite_1_yaw_rate*180./np.pi)
    ax[3, 1].scatter(flight_data.kite_actual_steering, flight_data.kite_1_yaw_rate*180./np.pi, s=.1)
    res = linregress(flight_data.kite_actual_steering, flight_data.kite_1_yaw_rate)
    ax[3, 1].plot(x, res.slope*x*180./np.pi, '--', color='C1')
    ax[3, 1].text(.05, .7, "r={:.2f}".format(res.rvalue), transform=ax[3, 1].transAxes)
    ax[3, 0].plot(flight_data.time, flight_data.kite_actual_steering*res.slope*180./np.pi, '--', color='C1')

    # ax[0].legend(bbox_to_anchor=(.15, 1.07, .7, .5), loc="lower left", mode="expand",
    #                borderaxespad=0, ncol=3)
    for a in ax.reshape(-1): a.grid()
    plot_flight_sections2(ax[:, 0], flight_data)
    add_panel_labels(np.delete(ax.reshape(-1), 1), [.26, .26, .38, .26, .38, .26, .38])
    ax[0, 0].get_shared_x_axes().join(*ax[:, 0])
    ax[0, 0].set_xlim([flight_data.iloc[0]['time'], flight_data.iloc[-1]['time']])
    ax[0, 1].get_shared_x_axes().join(*ax[:, 1])
    for a in ax.reshape(-1)[:6]: a.tick_params(labelbottom=False)


if __name__ == "__main__":
    plot_relations()  # Plots figure 11
    plt.show()