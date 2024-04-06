from sympy import symbols, diff, sin, cos, atan, acos
import numpy
from numpy import pi as num_pi

# коэффициенты
g = symbols('g')  # gamma
M_s = symbols('M_s')  # magnetiz sat
K1 = symbols('K1')  # anis const 1
K2 = symbols('K2')  # anis const 2
Nx = symbols('Nx')  # demag fact x
Ny = symbols('Ny')  # demag fact y
Nz = symbols('Nz')  # demag fact z
B1 = symbols('B1')  # magnetoelastic coef
B2 = symbols('B2')  # magnetoelastic coef
eps_xx = symbols('eps_xx')  # deformation
eps_yy = symbols('eps_yy')  # deformation
eps_zz = symbols('eps_zz')  # deformation
eps_xy = symbols('eps_xy')  # deformation
eps_xz = symbols('eps_xz')  # deformation
eps_yz = symbols('eps_yz')  # deformation

# компоненты H
Hx = symbols('Hx')
Hy = symbols('Hy')
Hz = symbols('Hz')

# переменные (углы в сферической системе координат)
x = symbols('x')  # тета
y = symbols('y')  # фи

# компоненты m
mx = sin(x) * cos(y)
my = sin(x) * sin(y)
mz = cos(x)

# -----------------------------------------выражение-для-энергии------------------------------------------------------ #

E_me = B1 * (mx ** 2 * eps_xx + my * eps_yy + mz * eps_zz) + \
       2 * B2 * (mx * my * eps_xy + mx * mz * eps_xz + my * mz * eps_yz)
E_ext = -M_s * (mx * Hx + my * Hy + mz * Hz)
E_ca = K1 * (mx ** 2 * my ** 2 + my ** 2 * mz ** 2 + mx ** 2 * mz ** 2) + \
       K2 * (mx ** 2 * my ** 2 * mz ** 2)
E_demag = M_s ** 2 / 2 * (Nx * mx ** 2 + Ny * my ** 2 + Nz * mz ** 2)

E = E_ext + E_demag + E_me + E_ca  # full energy

# первые производные по углам
fst_x_dif = diff(E, x)
fst_y_dif = diff(E, y)

# -----------------------------------------частота-прецессии---------------------------------------------------------- #

W_r = g / (M_s * sin(x)) * (diff(fst_x_dif, x) * diff(fst_y_dif, y) - (diff(fst_x_dif, y)) ** 2) ** 0.5

# перевод компонент
m_0 = [-1.0001, 0.0001, 0.0001]
m_0 = m_0 / numpy.linalg.norm(m_0)
num_x = acos(m_0[2])
num_y = atan(m_0[1] / m_0[0])

# значения коэффициентов, и переменных
values = {
    # постоянные 
    x: num_x, y: num_y,
    g: 1.76086 * 10 ** 7, M_s: 1707,
    K1: 4.2 * 10 ** 5, K2: 1.5 * 10 ** 5,
    B1: -2.9 * 10 ** 7, B2: 3.2 * 10 ** 7,
    
    # изменяемые
    Nx: 0, Ny: 0, Nz: 4 * num_pi,
    Hx: 3, Hy: 1, Hz: 2,
    eps_xx: 10**(-3), eps_yy: 10**(-3), eps_zz: 0, eps_xy: 0, eps_xz: 0, eps_yz: 0
}

print(W_r.subs(values))
