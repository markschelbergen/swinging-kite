import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.optimize import least_squares


def plot_vector(p0, v, ax, scale_vector=1, color='g', label=None):
    p1 = p0 + v * scale_vector
    vector = np.vstack(([p0], [p1])).T
    ax.plot3D(vector[0], vector[1], vector[2], color=color, label=label)


rho = 1.225
g = 9.81
d_t = .01
rho_t = 724.
cd_t = 1.1


def shoot(x, n_tether_elements, r_kite, v_kite, tension_ground, vwx, ax_plot_forces=False, return_positions=False):
    # Currently neglecting radial velocity of kite.
    beta_n, phi_n, tether_length = x
    om = np.cross(r_kite, v_kite)/np.linalg.norm(r_kite)**2
    l_s = tether_length/n_tether_elements
    m_s = np.pi*d_t**2/4 * l_s * rho_t

    tensions = np.zeros((n_tether_elements+1, 3))
    tensions[0, 0] = np.cos(beta_n)*np.cos(phi_n)*tension_ground
    tensions[0, 1] = np.sin(phi_n)*tension_ground
    tensions[0, 2] = np.sin(beta_n)*np.cos(phi_n)*tension_ground

    positions = np.zeros((n_tether_elements+1, 3))
    positions[1, 0] = np.cos(beta_n)*np.cos(phi_n)*l_s
    positions[1, 1] = np.sin(phi_n)*l_s
    positions[1, 2] = np.sin(beta_n)*np.cos(phi_n)*l_s

    for j in range(n_tether_elements):
        vj = np.cross(om, positions[j+1, :])  # TODO: only angular rotation considered
        aj = np.cross(om, vj)
        ej = (positions[j+1, :] - positions[j, :])/l_s  # Axial direction of tether element

        vaj = vj - np.array([vwx, 0, 0])  # Apparent wind velocity
        vajp = np.dot(vaj, ej)*ej  # Parallel to tether element
        vajn = vaj - vajp  # Perpendicular to tether element
        dj = -.5*rho*l_s*d_t*cd_t*np.linalg.norm(vajn)*vajn
        if j == n_tether_elements-1:
            m_s = .5*m_s
        fgj = np.array([0, 0, -m_s*g])
        tensions[j+1, :] = m_s*aj + tensions[j, :] - dj - fgj

        if ax_plot_forces:
            plot_vector(positions[j+1, :], dj, ax_plot_forces, color='r')
            plot_vector(positions[j+1, :], fgj, ax_plot_forces, color='k')
            plot_vector(positions[j+1, :], -m_s*aj, ax_plot_forces, color='m')
            plot_vector(positions[j+1, :], -tensions[j, :], ax_plot_forces, color='g')
            plot_vector(positions[j+1, :], tensions[j+1, :], ax_plot_forces, color='b')

        if j < n_tether_elements-1:
            positions[j+2, :] = positions[j+1, :] + tensions[j+1, :]/np.linalg.norm(tensions[j+1, :]) * l_s

    if return_positions:
        return positions
    else:
        return positions[-1, :] - r_kite


if __name__ == "__main__":
    args = (10, [200, 0, 100], [0, 20, 0], 1000, 9)
    opt_res = least_squares(shoot, (20*np.pi/180., -15*np.pi/180., 250), args=args, verbose=2)
    print(opt_res.x)
    p = shoot(opt_res.x, *args, return_positions=True)
    print(p[-1, :])

    plt.figure(figsize=(8, 6))
    ax3d = plt.axes(projection='3d')

    ax3d.plot(p[:, 0], p[:, 1], p[:, 2], 's')
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")

    ax3d.set_xlim([0, 250])
    ax3d.set_ylim([-125, 125])
    ax3d.set_zlim([0, 250])

    plt.show()
