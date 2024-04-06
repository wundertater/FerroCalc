import numpy as np
import matplotlib.pyplot as plt


def ferromag(init_cond: list | np.ndarray, start, stop, h, coefs: dict):
    """
    Решение уравнения LLG методом Рунге-Кутты 4 порядка
    :param init_cond: начальные условия
    :param start: начало отрезка интегрирования (начальное время)
    :param stop: конец отрезка интегрирования (конечное время)
    :param h: шаг решения
    :param coefs: коэффициенты уравнения:
    coefs = {a: float, H: np.ndarray, M_s: float, N_s: ndarray, k_1: float, k_2: float, B_1: float, B_2: float, eps: dict}
    (a - damping coefficient, H - внешнее поле, M_s - saturation magnetization, N_s - demagnetizing factors (x,y,z),
    k_1 - anisotropy constant первого порядка, k_2 - второго порядка, B_1 B_2 - magnetoelastic coupling constants,
    eps - деформации (если какие-то компоненты не заданы то равны 0))

    :return: (t_val, m_val)
    t_val - значения времени
    m_val - значения вектора m (матрица размера n x 3; первый столбец - компоненты x, второй - y, третий - z)
    m_val = [[m_x0, m_y0, m_z0],[m_x1, m_y1, m_z1], ..., [m_xn, m_yn, m_zn]]
    """
    # Коэффициенты
    g = 1.76086 * 10 ** 7  # gyromagnetic ratio of electron (gamma)
    a = coefs['a']  # damping coefficient
    H = coefs['H']
    M_s = coefs['M_s']
    N_s = np.array(coefs['N_s'])
    k_1 = coefs['k_1']
    k_2 = coefs['k_2']
    B_1 = coefs['B_1']
    B_2 = coefs['B_2']
    eps_xx = coefs['eps']['eps_xx'] if 'eps_xx' in coefs['eps'] else 0
    eps_yy = coefs['eps']['eps_yy'] if 'eps_yy' in coefs['eps'] else 0
    eps_zz = coefs['eps']['eps_zz'] if 'eps_zz' in coefs['eps'] else 0
    eps_xy = coefs['eps']['eps_xy'] if 'eps_xy' in coefs['eps'] else 0
    eps_xz = coefs['eps']['eps_xz'] if 'eps_xz' in coefs['eps'] else 0
    eps_yz = coefs['eps']['eps_yz'] if 'eps_yz' in coefs['eps'] else 0

    def energy(m):
        """
        Подсчет полной энергии
        """
        H_demag = - N_s * m * M_s
        mx2, my2, mz2 = m[0] ** 2, m[1] ** 2, m[2] ** 2  # Компоненты вектора m в квадрате

        E_ca = k_1 * (mx2 * my2 + my2 * mz2 + mx2 * mz2) + k_2 * (mx2 * my2 * mz2)  # Энергия кубической анизотропии
        E_demag = -0.5 * M_s * np.dot(m, H_demag)  # Энергия размагничивания
        E_ext = - M_s * np.dot(m, H)  # Энергия от внешнего поля
        E_me = B_1 * (mx2 * eps_xx + my2 * eps_yy + mz2 * eps_zz) + \
               2 * B_2 * (m[0] * m[1] * eps_xy + m[0] * m[2] * eps_xz + m[1] * m[2] * eps_yz)  # Магнитоупругая энергия

        return E_demag + E_me + E_ca + E_ext

    def llg_equation(m: np.ndarray):
        m = m / np.linalg.norm(m)  # доп. нормировка на входе (bug fixed)
        """
        Уравнение LLG
        :param m: вектор намагниченности
        """
        _energy = energy(m)
        H_eff_x = -(energy(m + [h, 0, 0]) - _energy) / (h * M_s)  # частные производные: dE/dm_x, dE/dm_y, dE/dm_z
        H_eff_y = -(energy(m + [0, h, 0]) - _energy) / (h * M_s)  # dE/dM = -1/M_s * dE/dm, M=M_s*m
        H_eff_z = -(energy(m + [0, 0, h]) - _energy) / (h * M_s)
        H_eff = [H_eff_x, H_eff_y, H_eff_z]  # полное магнитное поле

        mxH = np.cross(m, H_eff)  # первое слагаемое в скобках уравнения (m x H_eff)
        mx_mxH = np.cross(m, mxH)  # второе слагаемое (m x (m x H_eff))
        return (- g / (1 + a ** 2)) * (mxH + a * mx_mxH)

    n = int((stop - start) / h)  # количество разбиений
    t_val = np.zeros(n)
    t_val[0] = start
    m_val = np.zeros([n, 3])  # матрица векторов m
    init_cond = init_cond / np.linalg.norm(init_cond)  # нормализация начальных условий
    m_val[0] = init_cond

    for i in range(1, n):
        t = t_val[i - 1]
        m = m_val[i - 1]

        # Коэффициенты Рунге-Кутты
        k1 = h * llg_equation(m)
        k2 = h * llg_equation(m + k1 / 2)
        k3 = h * llg_equation(m + k2 / 2)
        k4 = h * llg_equation(m + k3)

        m_new = m + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        m_new = m_new / np.linalg.norm(m_new)  # нормализация |m|=1

        t_val[i] = t + h
        m_val[i] = m_new

    return t_val, m_val


if __name__ == '__main__':
    H = [3, 1, 2]
    m_0 = [0, 1, 1]
    m_0 = m_0 / np.linalg.norm(m_0)
    coefs = {'a': 0.1, 'H': H, 'M_s': 1707, 'N_s': [0, 0, 4 * np.pi],
             'k_1': 4.2 * 10 ** 5, 'k_2': 1.5 * 10 ** 5, 'B_1': -2.9 * 10 ** 7, 'B_2': 3.2 * 10 ** 7,
             'eps': {'eps_xx': -0.63, 'eps_yy': -0.63}}

    t, m = ferromag(m_0, 0, 10 ** (-7), 10 ** (-10), coefs)
    plt.plot(t, [m[i][0] for i in range(len(m))], label='m_x')
    plt.plot(t, [m[i][1] for i in range(len(m))], label='m_y')
    plt.plot(t, [m[i][2] for i in range(len(m))], label='m_z')
    m_0 = list(m_0)
    m_0[0], m_0[1], m_0[2] = round(m_0[0], 10), round(m_0[1], 10), round(m_0[2], 10)
    plt.title(f'{H=} {m_0=}')

    plt.legend()
    plt.grid()
    plt.savefig('data_llg/result.png')
    plt.show()
