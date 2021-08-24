import casadi as ca
from casadi.casadi import Opti
import numpy as np
import matplotlib.pyplot as plt
import pickle
from generate_initial_state import get_pyramid, get_tilted_pyramid, get_moving_pyramid, check_constraints

rho = 1.225
g = 9.81

d_t = .01
rho_t = 724.
cd_t = 1.1

m_kite = 11
m_kcu = 9
l_bridle = 11.5


def get_catenary_sag(x, tether_force_center):
    a = tether_force_center/(g*rho_t*np.pi*d_t**2/4)
    return a * np.cosh((x-x[-1]/2)/a) - a * np.cosh(x[-1]/2/a)


def derive_tether_model(n_elements, fix_end=False, explicit=True, include_drag=True, omega=None, vwx=0, impose_acceleration_directly=False):
    vw = ca.vertcat(vwx, 0, 0)

    # States
    if fix_end:
        n_free_point_masses = n_elements - 1
    else:
        n_free_point_masses = n_elements
    r = ca.SX.sym('r', n_free_point_masses, 3)
    v = ca.SX.sym('v', n_free_point_masses, 3)

    l = ca.SX.sym('l')
    dl = ca.SX.sym('dl')
    x = ca.vertcat(ca.vec(r.T), ca.vec(v.T), l, dl)

    # Controls
    ddl = ca.SX.sym('ddl')
    if impose_acceleration_directly:
        assert omega is None, "Setting omega already imposes the acceleration of the last point mass."
        a_end = ca.SX.sym('a_end', 3)
        u = ca.vertcat(ddl, a_end)
    else:
        u = ddl

    l_s = l / n_elements
    dl_s = dl / n_elements
    ddl_s = ddl / n_elements

    m_s = np.pi*d_t**2/4 * l_s * rho_t

    if include_drag:
        # Determine drag on each tether element
        d_s = []
        for i in range(n_elements):
            if i == 0:
                v0 = ca.SX.zeros((1, 3))
            else:
                v0 = v[i-1, :]
            if fix_end and i == n_free_point_masses:
                vf = ca.SX.zeros((1, 3))
            else:
                vf = v[i, :]

            va = (v0+vf)/2 - vw.T
            d_s.append(-.5*rho*d_t*l_s*cd_t*ca.norm_2(va)*va)
        d_s = ca.vcat(d_s)
    else:
        d_s = ca.SX.zeros(n_elements, 3)

    e_k = 0
    e_p = 0
    f = []
    tether_lengths = []
    tether_length_constraints = []

    for i in range(n_elements):
        last_element = i == n_elements - 1

        if fix_end and i == n_free_point_masses:
            if isinstance(fix_end, list):
                zi = fix_end[2]
            else:
                zi = 0
            vi = ca.SX.zeros((1, 3))
        else:
            zi = r[i, 2]
            vi = v[i, :]
        vai = vi - vw.T

        if last_element:
            point_mass = m_s/2 # + m_kite
        else:
            point_mass = m_s

        e_k = e_k + .5 * point_mass * ca.dot(vi, vi)
        if not (last_element and (omega is not None or impose_acceleration_directly)):
            e_p = e_p + point_mass * zi * g
        else:
            print("Resultant force on last mass point is imposed, i.e. gravity and tether forces etc. on last mass "
                  "point are not included explicitly.")

        if last_element:
            if omega is not None:
                fi = ca.cross(omega, vi)*point_mass  # Centripetal force w/o tether drag and aerodynamic force of kite.
            elif impose_acceleration_directly:
                fi = a_end.T*point_mass
            else:
                fi = d_s[i, :]/2  # + fa_kite
        else:
            fi = (d_s[i, :] + d_s[i+1, :])/2
        if not (fix_end and last_element):
            f.append(fi)

        if i == 0:
            ri0 = ca.SX.zeros((1, 3))
        else:
            ri0 = r[i-1, :]
        if fix_end and last_element:
            if isinstance(fix_end, list):
                rif = ca.hcat(fix_end)
            else:
                rif = ca.horzcat(fix_end, 0, 0)
        else:
            rif = r[i, :]
        dri = rif - ri0
        tether_lengths.append(dri)
        tether_length_constraints.append(.5 * (ca.dot(dri, dri) - l_s**2))

        if last_element:
            ez_end_sec = dri/ca.norm_2(dri)
            ey_end_sec = ca.cross(ez_end_sec, vai)/ca.norm_2(ca.cross(ez_end_sec, vai))
            ex_end_sec = ca.cross(ey_end_sec, ez_end_sec)
            dcm_end_sec = ca.horzcat(ex_end_sec.T, ey_end_sec.T, ez_end_sec.T)

            ez_tau = rif/ca.norm_2(rif)
            ey_tau = ca.cross(ez_tau, vai)/ca.norm_2(ca.cross(ez_tau, vai))
            ex_tau = ca.cross(ey_tau, ez_tau)
            dcm_tau = ca.horzcat(ex_tau.T, ey_tau.T, ez_tau.T)

    f = ca.vcat(f)
    tether_lengths = ca.vcat(tether_lengths)
    tether_length_constraints = ca.vcat(tether_length_constraints)

    r = ca.vec(r.T)
    v = ca.vec(v.T)
    f = ca.vec(f.T)

    a00 = ca.jacobian(ca.jacobian(e_k, v), v)
    a10 = ca.jacobian(tether_length_constraints, r)
    a0 = ca.horzcat(a00, a10.T)
    a1 = ca.horzcat(a10, ca.SX.zeros((n_elements, n_elements)))
    a = ca.vertcat(a0, a1)
    if not fix_end and (omega is not None or impose_acceleration_directly):  # Set tether force on last point mass to zero
        a[(n_elements - 1) * 3:n_elements * 3, -1] = 0

    c0 = f - ca.jacobian(e_p, r).T
    c1 = -ca.jacobian(a10@v, r)@v + (dl_s**2 + l_s*ddl_s) * ca.SX.ones(n_elements, 1)
    c = ca.vertcat(c0, c1)

    tether_speed_constraints = a10@v - l_s*dl_s * ca.SX.ones(n_elements, 1)

    if explicit:
        b = ca.mtimes(ca.inv(a), c)
        rhs = ca.vertcat(v, b[:3*n_free_point_masses], dl, ddl)
        nu = b[3*n_free_point_masses:]
        res = {
            'x': x,
            'u': u,
            'rhs': rhs,
            'nu': nu,
            'b': b,
            'n_elements': n_elements,
        }
    else:
        res = {
            'x': x,
            'u': u,
            'a': a,
            'c': c,
            'a10': a10,
            'g': tether_length_constraints,
            'dg': tether_speed_constraints,
            'n_free_pm': n_free_point_masses,
            'n_elements': n_elements,
            'tether_lengths': tether_lengths,
            'rotation_matrices': {
                'tangential_plane': dcm_tau,
                'last_element': dcm_end_sec,
            }
        }
    return res


def derive_tether_model_kcu(n_tether_elements, separate_kcu_mass=False, explicit=True, include_drag=True, vwx=0, impose_acceleration_directly=False):
    # Removed the functionality to fix the end and imposing a rigid body rotation with omega.
    vw = ca.vertcat(vwx, 0, 0)

    # States
    n_elements = n_tether_elements
    if separate_kcu_mass:
        n_elements += 1
    n_free_point_masses = n_elements
    r = ca.SX.sym('r', n_free_point_masses, 3)
    v = ca.SX.sym('v', n_free_point_masses, 3)

    l = ca.SX.sym('l')
    dl = ca.SX.sym('dl')
    x = ca.vertcat(ca.vec(r.T), ca.vec(v.T), l, dl)

    # Controls
    ddl = ca.SX.sym('ddl')
    if impose_acceleration_directly:
        a_end = ca.SX.sym('a_end', 3)
        u = ca.vertcat(ddl, a_end)
    else:
        u = ddl

    l_s = l / n_tether_elements
    dl_s = dl / n_tether_elements
    ddl_s = ddl / n_tether_elements

    m_s = np.pi*d_t**2/4 * l_s * rho_t

    if include_drag:
        # Determine drag on each tether element
        d_s = []
        for i in range(n_tether_elements):
            if i == 0:
                v0 = ca.SX.zeros((1, 3))
            else:
                v0 = v[i-1, :]
            vf = v[i, :]

            va = (v0+vf)/2 - vw.T
            d_s.append(-.5*rho*d_t*l_s*cd_t*ca.norm_2(va)*va)
        d_s = ca.vcat(d_s)
        if separate_kcu_mass:
            # TODO: add bridle drag?
            v_kcu = v[n_tether_elements-1, :]
            va = v_kcu - vw.T
            d_kcu = -.5*rho*1.6*1.*ca.norm_2(va)*va
    else:
        d_s = ca.SX.zeros(n_tether_elements, 3)

    e_k = 0
    e_p = 0
    f = []  # Non-conservative forces
    tether_lengths = []
    tether_length_constraints = []

    for i in range(n_elements):
        last_element = i == n_elements - 1
        kcu_element = separate_kcu_mass and i == n_elements - 2

        zi = r[i, 2]
        vi = v[i, :]
        vai = vi - vw.T

        if not separate_kcu_mass:
            if last_element:
                point_mass = m_s/2 + m_kite + m_kcu
            else:
                point_mass = m_s
        else:
            if last_element:
                point_mass = m_kite
            elif kcu_element:
                point_mass = m_s/2 + m_kcu
            else:
                point_mass = m_s

        e_k = e_k + .5 * point_mass * ca.dot(vi, vi)
        if last_element and impose_acceleration_directly:
            print("Resultant force on last mass point is imposed, i.e. gravity and tether forces etc. on last mass "
                  "point are not included explicitly.")
        else:
            e_p = e_p + point_mass * zi * g

        if last_element:
            if impose_acceleration_directly:
                fi = a_end.T*point_mass
            elif not separate_kcu_mass:
                fi = d_s[i, :]/2  # + fa_kite
            else:
                fi = 0  # fa_kite
        elif kcu_element:
            fi = d_s[i, :]/2 + d_kcu
        else:
            fi = (d_s[i, :] + d_s[i+1, :])/2
        f.append(fi)

        if i == 0:
            ri0 = ca.SX.zeros((1, 3))
        else:
            ri0 = r[i-1, :]
        rif = r[i, :]
        dri = rif - ri0
        tether_lengths.append(dri)

        if last_element and separate_kcu_mass:
            tether_length_constraints.append(.5 * (ca.dot(dri, dri) - l_bridle**2))
        else:
            tether_length_constraints.append(.5 * (ca.dot(dri, dri) - l_s**2))

        if last_element:
            ez_end_sec = dri/ca.norm_2(dri)
            ey_end_sec = ca.cross(ez_end_sec, vai)/ca.norm_2(ca.cross(ez_end_sec, vai))
            ex_end_sec = ca.cross(ey_end_sec, ez_end_sec)
            dcm_end_sec = ca.horzcat(ex_end_sec.T, ey_end_sec.T, ez_end_sec.T)

            ez_tau = rif/ca.norm_2(rif)
            ey_tau = ca.cross(ez_tau, vai)/ca.norm_2(ca.cross(ez_tau, vai))
            ex_tau = ca.cross(ey_tau, ez_tau)
            dcm_tau = ca.horzcat(ex_tau.T, ey_tau.T, ez_tau.T)

    f = ca.vcat(f)
    tether_lengths = ca.vcat(tether_lengths)
    tether_length_constraints = ca.vcat(tether_length_constraints)

    r = ca.vec(r.T)
    v = ca.vec(v.T)
    f = ca.vec(f.T)

    a00 = ca.jacobian(ca.jacobian(e_k, v), v)
    a10 = ca.jacobian(tether_length_constraints, r)
    a0 = ca.horzcat(a00, a10.T)
    a1 = ca.horzcat(a10, ca.SX.zeros((n_elements, n_elements)))
    a = ca.vertcat(a0, a1)
    if impose_acceleration_directly:  # Set tether force on last point mass to zero
        a[(n_elements - 1) * 3:n_elements * 3, -1] = 0

    c0 = f - ca.jacobian(e_p, r).T
    if separate_kcu_mass:
        c1 = -ca.jacobian(a10@v, r)@v + ca.vertcat((dl_s**2 + l_s*ddl_s) * ca.SX.ones(n_tether_elements, 1), 0)
    else:
        c1 = -ca.jacobian(a10@v, r)@v + (dl_s**2 + l_s*ddl_s) * ca.SX.ones(n_tether_elements, 1)
    c = ca.vertcat(c0, c1)

    if separate_kcu_mass:
        tether_speed_constraints = a10@v - ca.vertcat(l_s*dl_s * ca.SX.ones(n_tether_elements, 1), 0)
    else:
        tether_speed_constraints = a10@v - l_s*dl_s * ca.SX.ones(n_elements, 1)

    if explicit:
        b = ca.mtimes(ca.inv(a), c)
        rhs = ca.vertcat(v, b[:3*n_free_point_masses], dl, ddl)
        nu = b[3*n_free_point_masses:]
        res = {
            'x': x,
            'u': u,
            'rhs': rhs,
            'nu': nu,
            'b': b,
            'n_elements': n_elements,
        }
    else:
        res = {
            'x': x,
            'u': u,
            'a': a,
            'c': c,
            'a10': a10,
            'g': tether_length_constraints,
            'dg': tether_speed_constraints,
            'n_free_pm': n_free_point_masses,
            'n_elements': n_elements,
            'n_tether_elements': n_tether_elements,
            'tether_lengths': tether_lengths,
            'rotation_matrices': {
                'tangential_plane': dcm_tau,
                'last_element': dcm_end_sec,
            }
        }
    return res


def collocation_integration(dt, x0, control_input, model, nu_init=None):
    n_free_pm = model['n_free_point_masses']
    opti = Opti()

    # Dictionary collecting all decision variables.
    degree = 3
    method = 'radau'
    tau = ca.collocation_points(degree, method)
    c, d = ca.collocation_coeff(tau)[:2]

    z = opti.variable(model['n_states'], degree+1)
    x_init = x0[:]
    x_init[n_free_pm*3+2:n_free_pm*6:3] = x_init[n_free_pm*3+2:n_free_pm*6:3] + 1e-6
    opti.set_initial(z, np.repeat(x_init, degree+1, axis=1))
    xc = z[:, 1:]
    nuc = opti.variable(model['n_elements'], degree)
    if nu_init is not None:
        opti.set_initial(nuc, np.repeat(nu_init, degree, axis=1))

    d_pi = (z @ c)/dt  # Derivatives of polynomials at collocation time points.
    opti.subject_to(d_pi[:n_free_pm*3, :] == xc[n_free_pm*3:n_free_pm*6, :])  # Polynomial-actual trajectory consistency constraints.
    for j in range(degree):
        a, e = model['fun_mat'](xc[:, j], control_input)
        b = ca.vertcat(d_pi[n_free_pm*3:n_free_pm*6, j], nuc[:, j])
        opti.subject_to(a @ b == e)
    opti.subject_to(d_pi[n_free_pm*6, :] == xc[n_free_pm*6+1, :])
    opti.subject_to(d_pi[n_free_pm*6+1, :] == control_input[0])

    # Starting point constraints
    opti.subject_to(z[:, 0] == x0.reshape((-1, 1)))

    print_level = 5
    opti.solver('ipopt', {'ipopt': {'print_level': print_level, 'max_iter': 100, "linear_solver": "ma57"}})
    try:
        sol = opti.solve()
        get_values = sol.value
        opt_succeeded = True
    except RuntimeError as e:
        get_values = opti.debug.value
        opt_succeeded = False

    return opt_succeeded, get_values(z), get_values(nuc)


def find_steady_state(dyn, l, dl, x_init):
    # Used in static_state_analysis()
    opti = Opti()

    r_ss = opti.variable(dyn['n_free_pm']*3)
    opti.set_initial(r_ss, x_init[:dyn['n_free_pm']*3])
    v_ss = opti.variable(dyn['n_free_pm']*3)
    opti.set_initial(v_ss, x_init[dyn['n_free_pm']*3:dyn['n_free_pm']*6])
    ss = ca.vertcat(r_ss, v_ss, l, dl)

    nu = opti.variable(dyn['n_elements'])

    fun_mat = ca.Function('f_mat', [dyn['x'], dyn['u']], [dyn['a10'], dyn['c'], dyn['g'], dyn['dg']])
    a10, c, g, dg = fun_mat(ss, 0)
    opti.subject_to(a10.T @ nu == c[:dyn['n_free_pm']*3])
    opti.subject_to(c[dyn['n_free_pm']*3:] == 0)
    opti.subject_to(g == 0)
    opti.subject_to(dg == 0)

    print_level = 5
    opti.solver('ipopt', {'ipopt': {'print_level': print_level, 'max_iter': 100, "linear_solver": "ma57"}})
    try:
        sol = opti.solve()
        get_values = sol.value
        opt_succeeded = True
    except RuntimeError as e:
        get_values = opti.debug.value
        opt_succeeded = False

    return opt_succeeded, np.array(get_values(ss)), np.array(get_values(nu))


def ode_sim(tf, n_intervals, n_elements, fix_end, include_drag=True):
    dyn = derive_tether_model(n_elements, fix_end, include_drag=include_drag)
    fun_nu = ca.Function('f', [dyn['x'], dyn['u']], [dyn['nu'].T])
    ode = {'x': dyn['x'], 'ode': dyn['rhs'], 'p': dyn['u']}
    intg = ca.integrator('intg', 'cvodes', ode, {"tf": tf})
    # options = {
    #     "tf": tf,
    #     "number_of_finite_elements": 1,
    #     "interpolation_order": 3,
    #     "collocation_scheme": 'radau',
    # }
    # intg = ca.integrator('intg', 'collocation', ode, options)

    n_states = dyn['x'].shape[0]
    n_controls = dyn['u'].shape[0]
    x0 = ca.MX.sym('x0', n_states)
    u = ca.MX.sym('u', n_intervals, n_controls)

    x_sol = [x0.T]
    nu_sol = []
    for i in range(n_intervals):
        xf = intg(x0=x_sol[-1], p=u[i, :])["xf"].T
        x_sol.append(xf)

        nu_sol.append(fun_nu(x_sol[-1], u[i, :]))

    x_sol = ca.vcat(x_sol)
    nu_sol = ca.vcat(nu_sol)

    return ca.Function('sim', [x0, u], [x_sol, nu_sol])


def dae_sim(tf, n_intervals, dyn):
    # Returns nu at end of intervals
    a = ca.vec(ca.SX.sym('a', dyn['n_free_pm'], 3).T)
    nu = ca.SX.sym('nu', dyn['n_elements'])
    z = ca.vertcat(a, nu)

    f_z = dyn['a'] @ z - dyn['c']
    f_x = ca.vertcat(dyn['x'][dyn['n_free_pm']*3:dyn['n_free_pm']*6], a, dyn['x'][dyn['n_free_pm']*6+1], dyn['u'][0])

    # Create an integrator
    dae = {'x': dyn['x'], 'z': z, 'p': dyn['u'], 'ode': f_x, 'alg': f_z}

    # options = {
    #     "tf": tf,
    #     "number_of_finite_elements": 1,
    #     "interpolation_order": 3,
    #     "collocation_scheme": 'radau',
    # }
    # intg = ca.integrator('intg', 'collocation', dae, options)
    intg = ca.integrator('intg', 'idas', dae, {'tf': tf})

    n_states = dyn['x'].shape[0]
    n_controls = dyn['u'].shape[0]
    x0 = ca.MX.sym('x0', n_states)
    u = ca.MX.sym('u', n_intervals, n_controls)

    x_sol = [x0.T]
    nu_sol = []
    for i in range(n_intervals):
        sol = intg(x0=x_sol[-1], p=u[i, :])
        x_sol.append(sol["xf"].T)
        nu_sol.append(sol["zf"][dyn['n_free_pm']*3:].T)

    x_sol = ca.vcat(x_sol)
    nu_sol = ca.vcat(nu_sol)

    return ca.Function('sim', [x0, u], [x_sol, nu_sol])


def run_simulation(integrator=1):
    import time
    n_elements = 3
    animate = True
    catenary_validation = True
    reel_out_test = False
    fix_end = False  #60
    tf = .3  # Time step
    n_intervals = 100
    t = np.arange(0, n_intervals+1)*tf

    # Get starting position
    l0 = 75
    dl0 = 0
    assert dl0 == 0, "Current starting point only valid for zero reel-out speed"
    if fix_end:
        n_free_pm = n_elements-1
        r = np.zeros((n_free_pm, 3))
        x, z = get_pyramid(l0, fix_end, n_elements)
        r[:, 0] = x[:-1]
        r[:, 2] = z[:-1]
        v = np.zeros((n_free_pm, 3))
    else:
        n_free_pm = n_elements
        r = np.zeros((n_elements, 3))
        r[:, 0] = np.linspace(0, l0, n_elements+1)[1:]
        v = np.zeros((n_elements, 3))
    x0 = np.vstack((r.reshape((-1, 1)), v.reshape((-1, 1)), [[l0], [dl0]]))

    # Run simulation
    u = np.zeros((n_intervals, 1))
    if reel_out_test:
        u[10:20] = 1

    sol_nu = None
    if integrator == 0:
        dyn = derive_tether_model(n_elements, fix_end, False)
        fun_mat = ca.Function('f_mat', [dyn['x'], dyn['u']], [dyn['a'], dyn['c']])
        n_states = x0.shape[0]
        model = {'n_states': n_states, 'n_elements': n_elements, 'n_free_point_masses': n_free_pm, 'fun_mat': fun_mat}
        nu0 = None
        sol_x = x0.T

        start_time = time.monotonic()
        for k, uk in enumerate(u):
            success, sol_xk, sol_nuk = collocation_integration(tf, x0, uk, model, nu0)
            if not success:
                print("Failed at interval {}".format(str(k)))
                n_intervals = k
                break
            x0 = sol_xk[:, -1:]
            nu0 = sol_nuk[:, -1:]
            sol_x = np.vstack((sol_x, sol_xk[:, -1]))
        print('DAE time spent: ', time.monotonic() - start_time)
    elif integrator == 1:
        sim = ode_sim(tf, n_intervals, n_elements, fix_end)
        start_time = time.monotonic()
        res = sim(x0, u)
        print('ODE time spent: ', time.monotonic() - start_time)
        sol_x = np.array(res[0])
    elif integrator == 2:
        sim = ode_sim(1e-3, 1, n_elements, fix_end, include_drag=False)
        res = sim(x0, u[:1, :])
        x0 = np.array(res[0][1:, :].T)

        dyn = derive_tether_model(n_elements, fix_end, False)
        sim = dae_sim(tf, n_intervals, dyn)
        start_time = time.monotonic()
        sol_x, sol_nu = sim(x0, u)
        print('DAE time spent: ', time.monotonic() - start_time)
        sol_x = np.array(sol_x)
        sol_nu = np.array(sol_nu)

    if fix_end:
        rx = np.hstack((np.zeros((n_intervals+1, 1)), sol_x[:, :n_free_pm*3:3], np.ones((n_intervals+1, 1)) * fix_end))
        ry = np.hstack((np.zeros((n_intervals+1, 1)), sol_x[:, 1:n_free_pm*3:3], np.zeros((n_intervals+1, 1))))
        rz = np.hstack((np.zeros((n_intervals+1, 1)), sol_x[:, 2:n_free_pm*3:3], np.zeros((n_intervals+1, 1))))
    else:
        rx = np.hstack((np.zeros((n_intervals+1, 1)), sol_x[:, :n_elements*3:3]))
        ry = np.hstack((np.zeros((n_intervals+1, 1)), sol_x[:, 1:n_elements*3:3]))
        rz = np.hstack((np.zeros((n_intervals+1, 1)), sol_x[:, 2:n_elements*3:3]))

    if animate:
        ax = plt.figure().gca()
        for i in range(n_intervals+1):
            ax.cla()
            ax.set_xlim([-150, 150])
            ax.set_ylim([-150, 150])
            ax.plot(rx[i, :], rz[i, :])
            ax.text(75, 75, str(i))
            plt.pause(0.05)

    if sol_nu is None:
        return

    l = ((rx[:, 1:] - rx[:, :-1])**2 + (ry[:, 1:] - ry[:, :-1])**2 + (rz[:, 1:] - rz[:, :-1])**2)**.5
    tether_force = sol_nu*l[1:, :]

    plt.figure()
    plt.plot(t[1:], tether_force)
    plt.ylabel("Tether force [N]")

    if fix_end and catenary_validation:
        assert n_elements % 2 == 1
        tether_force_center = np.mean(tether_force[-100:, (n_elements-1)//2])

        plt.figure()
        plt.plot(rx[-1, :], rz[-1, :])
        x = np.linspace(0, fix_end)
        y = get_catenary_sag(x, tether_force_center)
        plt.plot(x, y, '--')

    dyn = derive_tether_model(n_elements, fix_end, False)
    get_a10 = ca.Function('get_a10', [dyn['x'], dyn['u']], [dyn['a10']])
    dl = np.empty((0, n_elements))
    for xk, uk, rxk, ryk, rzk in zip(sol_x[1:, :], u, rx[1:, :], ry[1:, :], rz[1:, :]):
        l_s = ((rxk[1:] - rxk[:-1])**2 + (ryk[1:] - ryk[:-1])**2 + (rzk[1:] - rzk[:-1])**2)**.5
        a10k = np.array(get_a10(xk, uk))
        vk = xk[n_free_pm*3:n_free_pm*6]
        dl = np.vstack((dl, a10k@vk/l_s))

    # Especially interesting for reel out test
    fig, ax = plt.subplots(2, 2)
    plt.suptitle("Constraint check")
    ax[0, 0].set_title("Tether length")
    ax[0, 0].plot(t, np.sum(l, axis=1), label='sum')
    ax[0, 0].plot(t, sol_x[:, n_free_pm*6], '--', label='state')
    ax[1, 0].set_title("Tether element length")
    ax[1, 0].plot(t, l)
    ax[1, 0].set_xlabel('Time [s]')

    ax[0, 1].set_title("Tether speed")
    ax[0, 1].plot(t[1:], np.sum(dl, axis=1), label='sum')
    ax[0, 1].plot(t, sol_x[:, n_free_pm*6+1], '--', label='state')
    ax[0, 1].legend()
    ax[1, 1].set_title("Tether element speed")
    ax[1, 1].plot(t[1:], dl)
    ax[1, 1].set_xlabel('Time [s]')


def static_state_analysis():
    n_elements = 5
    # Getting appropriate guess for the steady state, finding the exact steady state, and plotting the results.
    l = 75
    dl = 0
    fix_end = 60

    # Construct starting point
    if fix_end:
        n_free_pm = n_elements-1
        r = np.zeros((n_free_pm, 3))
        x, z = get_pyramid(l, fix_end, n_elements)
        r[:, 0] = x[:-1]
        r[:, 2] = z[:-1]
        v = np.zeros((n_free_pm, 3))
    else:
        n_free_pm = n_elements
        r = np.zeros((n_elements, 3))
        theta = 80*np.pi/180.
        r[:, 0] = np.linspace(0, l, n_elements+1)[1:]*np.sin(theta)
        r[:, 2] = np.linspace(0, l, n_elements+1)[1:]*-np.cos(theta)
        v = np.zeros((n_elements, 3))
    x0 = np.vstack((r.reshape((-1, 1)), v.reshape((-1, 1)), [[l], [dl]]))

    # Simulation can not cope with starting state without velocity when drag is included, therefore first let the model
    # run shortly without drag.
    sim = ode_sim(1e-3, 1, n_elements, fix_end, include_drag=False)
    res = sim(x0, 0)
    x0 = np.array(res[0][1:, :].T)

    # Next, run the model with drag for a bit longer to get a better approximation of the steady state.
    n_intervals = 10
    dyn = derive_tether_model(n_elements, fix_end, False)
    sim = dae_sim(.3, n_intervals, dyn)
    u = np.zeros((n_intervals, 1))
    sol_x, sol_nu = sim(x0, u)
    x0 = np.array(sol_x)[-1, :]

    # Now, find the steady state using the final state of the latter simulation.
    success, sol_ss, sol_nu = find_steady_state(dyn, l, dl, x0)
    print("Point mass positions:")
    print(sol_ss[:n_free_pm*3].reshape((-1, 3)))
    print("Point mass speeds:")
    print(sol_ss[n_free_pm*3:n_free_pm*6].reshape((-1, 3)))

    # Plot the results
    if fix_end:
        rx = np.hstack(([0], sol_ss[:n_free_pm*3:3], [fix_end]))
        rz = np.hstack(([0], sol_ss[2:n_free_pm*3:3], [0]))
    else:
        rx = np.hstack(([0], sol_ss[:n_free_pm*3:3]))
        rz = np.hstack(([0], sol_ss[2:n_free_pm*3:3]))
    plt.plot(rx, rz)

    if fix_end:
        if n_elements % 2 == 1:
            l_s = l/n_elements
            tether_force_center = sol_nu[(n_elements-1)//2]*l_s
            x = np.linspace(0, fix_end)
            y = get_catenary_sag(x, tether_force_center)
            plt.plot(x, y, '--')

        export_dict = {'x': sol_ss, 'nu': sol_nu, 'tether_length': l, 'fix_end': fix_end, 'n_elements': n_elements}
        with open("catenary.pickle", 'wb') as f:
            pickle.dump(export_dict, f)

    plt.xlim([-150, 150])
    plt.ylim([-150, 150])


def find_quasi_steady_state_fixed_end(dyn, omega, l, dl, fix_end_in_derivation, r_init):
    # For now no tether speed
    n_pm_opt = dyn['n_elements']-1

    opti = Opti()

    r_qss = opti.variable(n_pm_opt*3)
    v_qss = opti.variable(n_pm_opt*3)
    a_qss = opti.variable(n_pm_opt*3)

    if fix_end_in_derivation:
        qss = ca.vertcat(r_qss, v_qss, l, dl)
        kinematics_end = {}
    else:
        r_end = r_init[n_pm_opt*3:]
        v_end = np.cross(omega, r_end)
        qss = ca.vertcat(r_qss, r_end, v_qss, v_end, l, dl)
        a_end = np.cross(omega, v_end)
        kinematics_end = {
            'r': r_end,
            'v': v_end,
            'a': a_end,
        }

    # Determine and set initial guess
    opti.set_initial(r_qss, r_init[:n_pm_opt*3])
    v_init = np.empty((n_pm_opt, 3))
    a_init = np.empty((n_pm_opt, 3))

    r_init = r_init.reshape((-1, 3))
    for i in range(n_pm_opt):
        ri = r_init[i, :]
        v_init[i, :] = np.cross(omega, ri)
        a_init[i, :] = np.cross(omega, v_init[i, :])
    opti.set_initial(v_qss, v_init.reshape(-1))
    opti.set_initial(a_qss, a_init.reshape(-1))

    nu = opti.variable(dyn['n_elements'])

    fun_mat = ca.Function('f_mat', [dyn['x'], dyn['u']], [dyn['a'], dyn['c'], dyn['g'], dyn['dg']])
    a, c, g, dg = fun_mat(qss, 0)

    if fix_end_in_derivation:
        b = ca.vertcat(a_qss, nu)
    else:
        # Impose zero forces on last point mass
        c[n_pm_opt*3:(n_pm_opt+1)*3] = 0  # Set external forces to zero
        # Set mass and tether force on last point mass to zero
        a[n_pm_opt*3:(n_pm_opt+1)*3, :] = 0
        b = ca.vertcat(a_qss, a_end, nu)

    opti.subject_to(a @ b == c)

    opti.subject_to(g == 0)
    opti.subject_to(dg == 0)  # Might be trivial?

    opti.subject_to(nu > 0)

    r_all = ca.vertcat([0]*3, r_qss)
    if not fix_end_in_derivation:
        r_all = ca.vertcat(r_all, r_end)
    opti.subject_to((r_all[5::3]-r_all[2:-3:3])**2 < (l/dyn['n_elements'])**2)  # No vertical tether elements.

    velocity_errors = []
    acceleration_errors = []
    for i in range(n_pm_opt):
        ri = r_qss[i*3:(i+1)*3]
        vi = v_qss[i*3:(i+1)*3]
        ai = a_qss[i*3:(i+1)*3]

        vi_rot = ca.cross(omega, ri)
        # opti.subject_to(vi == vi_rot)  # + vi_trans)
        velocity_errors.append(vi - vi_rot)  # Seems to converge better when adding as objective.

        ai_rot = ca.cross(omega, ca.cross(omega, ri))
        # opti.subject_to(ai[0] == ai_rot[0])
        acceleration_errors.append(ai - ai_rot)
    velocity_errors = ca.vcat(velocity_errors)
    acceleration_errors = ca.vcat(acceleration_errors)

    objective_function = ca.sumsqr(acceleration_errors) + ca.sumsqr(velocity_errors)
    opti.minimize(objective_function)

    print_level = 5
    opti.solver('ipopt', {'ipopt': {'print_level': print_level, 'max_iter': 500, "linear_solver": "ma57"}})
    try:
        sol = opti.solve()
        get_values = sol.value
        opt_succeeded = True
    except RuntimeError as e:
        get_values = opti.debug.value
        opt_succeeded = False

    return opt_succeeded, np.array(get_values(r_qss)), np.array(get_values(v_qss)), np.array(get_values(a_qss)), \
           np.array(get_values(nu)), kinematics_end


def quasi_steady_state_analysis_fixed_end(print_solution=False):
    n_elements = 30

    # For 'free' end:
    dl = 0
    # Note that omega should be consistent with model, e.g.: fixed end on the x-axis only allow omega to have a
    # x-component.
    omega = np.array([0, 0, .8])
    fix_end_in_derivation = False

    l = 65
    fix_end_coords = [-20, 0, 50]
    print("Min tether length:", np.sum(np.array(fix_end_coords)**2)**.5)
    z, x = get_tilted_pyramid(l, fix_end_coords[2], fix_end_coords[0], n_elements)
    n_free_pm = n_elements-1

    r = np.zeros((n_free_pm, 3))
    r[:, 0] = x[:-1]
    r[:, 2] = z[:-1]
    r = r.reshape(-1)

    if not fix_end_in_derivation:
        r = np.hstack((r, fix_end_coords))
    else:
        fix_end_in_derivation = fix_end_coords

    dyn = derive_tether_model(n_elements, fix_end_in_derivation, False)
    success, sol_r, sol_v, sol_a, sol_nu, kinematics_end = find_quasi_steady_state_fixed_end(dyn, omega, l, dl, fix_end_in_derivation, r)
    eval_quasi_steady_state_fixed_end(dyn, omega, l, dl, fix_end_in_derivation, sol_r, sol_v, sol_a, sol_nu, kinematics_end)
    sol_r = sol_r.reshape((-1, 3))

    if print_solution:
        np.set_printoptions(suppress=True)
        print("Point mass positions:")
        print(sol_r)
        print("Point mass speeds:")
        print(sol_v.reshape((-1, 3)))
        print("Point mass accelerations:")
        print(sol_a.reshape((-1, 3)))
        print("Tether lengths:")
        l_s = np.sum((sol_r[1:, :] - sol_r[:-1, :])**2, axis=1)**.5
        print(l_s, np.sum(l_s))
        print("Langrangian multipliers:")
        print(sol_nu)

    # Plot the results
    rx = np.hstack(([0], sol_r[:, 0], [fix_end_coords[0]]))
    ry = np.hstack(([0], sol_r[:, 1], [fix_end_coords[1]]))
    rz = np.hstack(([0], sol_r[:, 2], [fix_end_coords[2]]))

    # print((rx[1:]-rx[:-1])/(l/n_elements))
    # print((ry[1:]-ry[:-1])/(l/n_elements))
    # print((rz[1:]-rz[:-1])/(l/n_elements))

    plt.plot(rx, rz, 's-')
    plt.plot(ry, rz, 's-')
    plt.plot(np.hstack(([0], r[::3])), np.hstack(([0], r[2::3])), '--')
    plt.axis('equal')

    plt.figure()
    plt.plot(rx, ry)
    plt.axis('equal')

    # plt.xlim([-150, 150])
    # plt.ylim([-150, 150])


def eval_quasi_steady_state_fixed_end(dyn, omega, l, dl, fix_end_in_derivation, r_qss, v_qss, a_qss, nu, kinematics_end):
    # For now no tether speed
    n_pm_opt = dyn['n_elements']-1

    if fix_end_in_derivation:
        qss = ca.vertcat(r_qss, v_qss, l, dl)
    else:
        qss = ca.vertcat(r_qss, kinematics_end['r'], v_qss, kinematics_end['v'], l, dl)

    fun_mat = ca.Function('f_mat', [dyn['x'], dyn['u']], [dyn['a'], dyn['c'], dyn['g'], dyn['dg'], dyn['a10']])
    a, c, g, dg, a10 = fun_mat(qss, 0)

    if fix_end_in_derivation:
        b = ca.vertcat(a_qss, nu)
    else:
        c[n_pm_opt*3:(n_pm_opt+1)*3] = 0
        a[n_pm_opt*3:(n_pm_opt+1)*3, :] = 0
        b = ca.vertcat(a_qss, kinematics_end['a'], nu)

    np.set_printoptions(precision=3, suppress=True, linewidth=np.nan)

    # print(np.array(a))
    # print(np.array(b))
    # print(np.array(c))

    constraints = {
        'dyn': a @ b - c,
        'g': g,
        'dg': dg,
    }

    for i in range(n_pm_opt):
        ri = r_qss[i*3:(i+1)*3]
        vi = v_qss[i*3:(i+1)*3]
        ai = a_qss[i*3:(i+1)*3]

        vi_rot = ca.cross(omega, ri)
        constraints['v{}'.format(str(i))] = vi - vi_rot  # + vi_trans)
        constraints['a{}'.format(str(i))] = ai - ca.cross(omega, ca.cross(omega, ri))

    for k, v in constraints.items():
        constraints[k] = np.array(v)
        print(k, v)

    constraints = np.vstack(list(constraints.values()))
    max_constraint = np.amax(np.abs(constraints))
    print("Max constraint:", max_constraint)
    # assert max_constraint < 1e-6


def run_simulation_dae():
    # dae part of run_simulation()
    n_elements = 30
    start_position_end = [-20, 0, 50]  #False  #60 #-10, 0, 60]
    omega = np.array([[0, 0, .8]])
    angular_speed = np.linalg.norm(omega)
    sim_periods = 2
    n_intervals = 100
    sim_time = 2*np.pi*sim_periods/angular_speed
    tf = sim_time/n_intervals  # Time step
    print("Simulation time and time step: {:.1f} s and {:.2f} s.".format(sim_time, tf))
    t = np.arange(0, n_intervals+1)*tf

    # Get starting position
    l0 = 65
    dl0 = 1

    # r = np.zeros((n_point_masses, 3))
    # r[:, 0] = np.linspace(0, l0, n_elements+1)[1:]
    # v = np.zeros((n_point_masses, 3))
    # v[:, 2] = np.linspace(0, .1, n_elements+1)[1:]
    #
    # x0 = np.vstack((r.reshape((-1, 1)), v.reshape((-1, 1)), [[l0], [dl0]]))

    vz_end = (start_position_end[2]/l0)**.5 * dl0
    print("Expected end height: {:.2f} m".format(vz_end*sim_time+start_position_end[2]))
    z, x, vz, vx = get_moving_pyramid(l0, start_position_end[2], start_position_end[0], dl0, vz_end, n_elements)
    n_free_pm = n_elements-1

    r = np.zeros((n_free_pm+1, 3))
    r[:, 0] = x
    r[:, 2] = z

    v = []
    for ri in r:
        v.append(np.cross(omega, ri))
    v = np.vstack(v)
    v[:, 0] = vx
    v[:, 2] = vz

    x0 = np.vstack((r.reshape((-1, 1)), v.reshape((-1, 1)), [[l0], [dl0]]))

    # Run simulation
    u = np.zeros((n_intervals, 1))

    if isinstance(start_position_end, list):
        fix_end_in_derivation = False
    else:
        fix_end_in_derivation = start_position_end
    dyn = derive_tether_model(n_elements, fix_end_in_derivation, False, omega=omega)
    check_constraints(dyn, x0)

    sim = dae_sim(tf, n_intervals, dyn)
    sol_x, sol_nu = sim(x0, u)
    sol_x = np.array(sol_x)
    sol_nu = np.array(sol_nu)

    rx = np.hstack((np.zeros((n_intervals+1, 1)), sol_x[:, :n_elements*3:3]))
    ry = np.hstack((np.zeros((n_intervals+1, 1)), sol_x[:, 1:n_elements*3:3]))
    rz = np.hstack((np.zeros((n_intervals+1, 1)), sol_x[:, 2:n_elements*3:3]))

    # plt.title("Radial position of tether free end")
    # plt.plot((rx[:, -1]**2+rz[:, -1]**2)**.5)
    # print("Final radius", (rx[-1, -1]**2+rz[-1, -1]**2)**.5)

    ax = plt.figure().gca()
    for i in range(n_intervals+1):
        ax.cla()
        ax.set_xlim([-50, 50])
        ax.set_ylim([-20, 80])
        ax.plot(rx[i, :], rz[i, :])
        ax.text(40, 60, str(i))
        plt.pause(0.05)

    plt.figure()
    plt.suptitle("Final position")
    plt.plot(rx[-1, :], rz[-1, :], 's-')
    plt.plot(ry[-1, :], rz[-1, :], 's-')
    plt.plot(rx[0, :], rz[0, :], '--')
    plt.plot(ry[0, :], rz[0, :], '--')
    plt.axis('equal')

    plt.figure()
    plt.suptitle("Final position top-view")
    plt.plot(rx[-1, :], ry[-1, :])
    plt.plot(rx[0, :], ry[0, :], '--')
    plt.axis('equal')

    plt.show()


def run_simulation_skip_rope():
    assert cd_t == 0
    assert g == 0
    # experimenting with rotating rope
    fix_end = 60
    tf = .1  # Time step
    n_intervals = 100
    omega = np.array([10, 0, 0])
    t = np.arange(0, n_intervals+1)*tf

    # Get starting position
    with open("catenary.pickle", 'rb') as f:
        catenary_sol = pickle.load(f)
    n_elements = catenary_sol['n_elements']
    n_free_pm = catenary_sol['n_elements']-1
    l = catenary_sol['tether_length']
    r = catenary_sol['x'][:n_free_pm*3]

    v = np.empty((n_free_pm, 3))
    for i, ri in enumerate(r.reshape((-1, 3))):
        v[i, :] = np.cross(omega, ri)

    x0 = np.vstack((r.reshape((-1, 1)), v.reshape((-1, 1)), [[l], [0]]))

    # Run simulation
    u = np.zeros((n_intervals, 1))

    dyn = derive_tether_model(n_elements, fix_end, False)
    sim = dae_sim(tf, n_intervals, dyn)
    sol_x, sol_nu = sim(x0, u)
    sol_x = np.array(sol_x)
    sol_nu = np.array(sol_nu)

    rx = np.hstack((np.zeros((n_intervals+1, 1)), sol_x[:, :n_free_pm*3:3], np.ones((n_intervals+1, 1)) * fix_end))
    ry = np.hstack((np.zeros((n_intervals+1, 1)), sol_x[:, 1:n_free_pm*3:3], np.zeros((n_intervals+1, 1))))
    rz = np.hstack((np.zeros((n_intervals+1, 1)), sol_x[:, 2:n_free_pm*3:3], np.zeros((n_intervals+1, 1))))

    from mpl_toolkits import mplot3d
    plt.figure(figsize=(12, 12))
    ax3d = plt.axes(projection='3d')

    ax = plt.figure().gca()

    for i in range(n_intervals+1):
        ax3d.cla()
        ax3d.set_xlim([-10, 70])
        ax3d.set_ylim([-50, 50])
        ax3d.set_zlim([-50, 50])
        ax3d.plot3D(rx[i, :], ry[i, :], rz[i, :])
        ax3d.text(75, 75, 75, str(i))

        ax.cla()
        ax.set_xlim([-10, 70])
        ax.set_ylim([-50, 50])
        ax.plot(rx[i, :], (ry[i, :]**2+rz[i, :]**2)**.5)
        ax.text(60, 40, str(i))

        plt.pause(0.1)

    plt.show()



if __name__ == "__main__":
    run_simulation()
    # static_state_analysis()
    # quasi_steady_state_analysis_fixed_end()
    # run_simulation_dae()
    # run_simulation_skip_rope()
    plt.show()
