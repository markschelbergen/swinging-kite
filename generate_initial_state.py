import numpy as np
import casadi as ca


def check_constraints(dyn, x0):
    cons_x0 = ca.Function('f', [dyn['x']], [dyn['g'], dyn['dg']])(x0)
    max_g = np.amax(np.abs(np.array(cons_x0[0])))
    print("Max. tether length constraints: {:.2E}".format(max_g))
    max_dg = np.amax(np.abs(np.array(cons_x0[1])))
    print("Max. tether speed constraints: {:.2E}".format(max_dg))
    assert max_g < 1e-5
    assert max_dg < 1e-5


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
