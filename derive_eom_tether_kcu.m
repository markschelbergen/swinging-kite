%%
clc;
clear all;

n_tether_elements = 1;
n_point_masses = n_tether_elements+1;

vwx = sym('vwx');
vw = [vwx; 0; 0];
rho = sym('rho');
g = sym('g');

m_k = sym('m_k');
m_kcu = sym('m_kcu');
l_bridle = sym('l_bridle');

l_t = sym('l_t');
dl_t = sym('dl_t');
ddl_t = sym('ddl_t');

l_s = l_t/n_tether_elements;
dl_s = dl_t/n_tether_elements;
ddl_s = ddl_t/n_tether_elements;

d_t = sym('d_t');
rho_t = sym('rho_t');
cd_t = sym('cd_t');

m_s = sym('m_s');  %pi*d_t^2/4 * l_s * rho_t;

fa = sym('fa', [1 3]);

% States
r = sym('r', [n_point_masses, 3]);
v = sym('v', [n_point_masses, 3]);

% Determine drag on each tether element
d_s = [];
for i = 1:n_tether_elements
    if i == 1
        v0 = zeros([1 3]);
    else
        v0 = v(i-1, :);
    end
    vf = v(i, :);
    
    va = (v0+vf)/2 - vw.';
    d_s = [d_s; -.5*rho*d_t*l_s*cd_t*norm(va)*va];
end
v_kcu = v(n_tether_elements, :);
va = v_kcu - vw.';
d_kcu = -.5*rho*1.6*1.*norm(va)*va;

e_k = 0;
e_p = 0;
f = [];
c = [];

for i = 1:n_point_masses
    zi = r(i, 3);
    vi = v(i, :);
    
    if i == n_point_masses
        point_mass = m_k;
    elseif i == n_point_masses-1
        point_mass = m_s/2 + m_kcu;
    else
        point_mass = m_s;
    end
    
    e_k = e_k + .5 * point_mass * vi * vi.';
    e_p = e_p + point_mass * zi * g;
    
    if i == n_point_masses
        fi = fa;
    elseif i == n_point_masses-1
        fi = d_s(i, :)/2 + d_kcu;
    else
        fi = (d_s(i, :) + d_s(i+1, :))/2;
    end
    f = [f; fi];
    
    if i == 1
        ri0 = zeros([1 3]);
    else
        ri0 = r(i-1, :);
    end
    rif = r(i, :);
    
    if i == n_point_masses
        ci = .5 * ((rif - ri0) * (rif - ri0).' - l_bridle^2);
    else
        ci = .5 * ((rif - ri0) * (rif - ri0).' - l_s^2);
    end
    c = [c; ci];
end

r = r.'; r = r(:);
v = v.'; v = v(:);
f = f.'; f = f(:);

a00 = jacobian(jacobian(e_k, v), v)
a10 = jacobian(c, r)

a = [a00, a10.'; a10, zeros(n_point_masses, n_point_masses)]
a((n_point_masses - 1) * 3+1:n_point_masses * 3+1, n_point_masses * 4) = 0

b0 = f - jacobian(e_p, r).'
b1 = -jacobian(a10*v, r)*v + [(dl_s^2 + l_s*ddl_s) * ones(n_tether_elements, 1); 0]
dl_s = a10*v/l_s

