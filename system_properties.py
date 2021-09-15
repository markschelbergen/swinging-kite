from numpy import pi

rho = 1.225
g = 9.81
d_t = .01
rho_t = 724.
cd_t = 1.1
tether_modulus = 614600/(pi*.002**2)  # From Uwe's thesis
tether_stiffness = tether_modulus*pi*(d_t/2)**2
# tether_stiffness = 490000

m_kite = 11
m_kcu = 16
l_bridle = 11.5
