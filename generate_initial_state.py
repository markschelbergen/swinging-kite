import numpy as np
import casadi as ca
from system_properties import l_bridle


def check_consistency_conditions(dyn, x):
    cond_x = ca.Function('f', [dyn['x']], [dyn['g'], dyn['dg']])(x)
    max_g = np.amax(np.abs(np.array(cond_x[0])))
    print("Max. tether length consistency conditions: {:.2E}".format(max_g))
    max_dg = np.amax(np.abs(np.array(cond_x[1])))
    print("Max. tether speed consistency conditions: {:.2E}".format(max_dg))
    assert max_g < 1e-5
    assert max_dg < 1e-5


def check_tether_element_length_drift(dyn, sol_x):
    import matplotlib.pyplot as plt

    fun_tether_element_lengths = ca.Function('f', [dyn['x']], [dyn['tether_lengths'], dyn['g'], dyn['dg']])
    tether_element_lengths = np.empty((sol_x.shape[0], dyn['n_elements']))
    tether_element_length_consistency = np.empty((sol_x.shape[0], dyn['n_elements']))
    tether_element_length_consistency_perc = np.empty((sol_x.shape[0], dyn['n_elements']))
    tether_element_velocity_consistency = np.empty((sol_x.shape[0], dyn['n_elements']))
    for i, x in enumerate(sol_x):
        l, c, cv = fun_tether_element_lengths(x)
        tether_element_lengths[i, :] = np.array(l).reshape(-1)
        tether_element_length_consistency[i, :] = np.array(c).reshape(-1)
        tether_element_length_consistency_perc[i, :-1] = tether_element_length_consistency[i, :-1]/(x[-2]/dyn['n_tether_elements'])*100
        tether_element_length_consistency_perc[i, -1] = tether_element_length_consistency[i, -1]/l_bridle*100
        tether_element_velocity_consistency[i, :] = np.array(cv).reshape(-1)

    fig, ax = plt.subplots(4, 1, sharex=True)
    ax[0].plot(tether_element_lengths)
    ax[0].plot(sol_x[:, -2]/dyn['n_tether_elements'], ls=':')
    ax[0].axhline(l_bridle, ls=':')
    ax[1].plot(tether_element_length_consistency)
    ax[2].plot(tether_element_length_consistency_perc)
    ax[3].plot(tether_element_velocity_consistency)


def find_initial_velocities_satisfying_constraints(dyn, x0, v_end, plot=False):
    n_point_masses = (len(x0)-2)//6
    n_pm_opt = n_point_masses-1
    r_guess = x0[:n_pm_opt*3]
    r_end = x0[n_pm_opt*3:n_point_masses*3]
    v_guess = x0[n_point_masses*3:2*n_point_masses*3-3]
    l = x0[2*n_point_masses*3]
    dl = x0[2*n_point_masses*3+1]

    opti = ca.casadi.Opti()
    # r = opti.variable(n_pm_opt*3)
    # opti.set_initial(r, r_guess)
    v = opti.variable(n_pm_opt*3)
    opti.set_initial(v, v_guess)

    x = ca.vertcat(r_guess, r_end, v, v_end, l, dl)
    g, dg = ca.Function('f', [dyn['x']], [dyn['g'], dyn['dg']])(x)
    opti.minimize(ca.sumsqr(g)+ca.sumsqr(dg))

    opti.solver('ipopt')
    sol = opti.solve()
    x_sol = sol.value(x)

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 12))
        ax3d = plt.axes(projection='3d')
        ax3d.set_xlim([0, 250])
        ax3d.set_ylim([-125, 125])
        ax3d.set_zlim([0, 250])
        ax3d.plot3D(x_sol[:3*n_point_masses:3], x_sol[1:3*n_point_masses:3], x_sol[2:3*n_point_masses:3], 's-')
        plt.show()
        exit()

    return x_sol
