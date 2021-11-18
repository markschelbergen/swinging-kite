from numpy import pi

vwx = 10

rho = 1.225
g = 9.81
d_t = .01
rho_t = 724.
cd_t = 1.1
tether_modulus = 614600/(pi*.002**2)  # From Uwe's thesis
tether_stiffness = tether_modulus*pi*(d_t/2)**2

m_kite = 11
m_kcu = 16
# m_kcu = 8
l_bridle = 11.5

cd_kcu = 1.
# frontal_area_kcu = 1.6
frontal_area_kcu = .25
