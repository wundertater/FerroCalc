import numpy as np
import matplotlib.pyplot as plt


def ferromag(init_cond: list | np.ndarray, start, stop, a: float, M_s: float,
             E_ext: dict = None, E_ca: dict = None, E_demag: dict = None, E_me: dict = None, h=None):
    """
    Решение уравнения LLG методом Рунге-Кутты 4 порядка для ферромагнетика
    :param init_cond: начальные условия
    :param start: начало отрезка интегрирования (начальное время)
    :param stop: конец отрезка интегрирования (конечное время)
    :param a: damping coefficient (from 0 to 1)
    :param M_s: saturation magnetization
    :param E_ext: энергия внешнего поля
    :param E_ca: энергия кубической анизотропии
    :param E_demag: энергия размагничивания
    :param E_me: магнитоупругая энергия
    :param h: шаг решения (по умолчанию: (stop - start) / 1000)
    :return: (t_val, m_val)
    t_val - значения времени
    m_val - значения вектора m (матрица размера n x 3; первый столбец - компоненты x, второй - y, третий - z)
    m_val = [[m_x0, m_y0, m_z0],[m_x1, m_y1, m_z1], ..., [m_xn, m_yn, m_zn]]

    ---coefficients for energies---
    * E_ext: {'H'}
        H: (list | ndarray) - внешнее поле
    * E_ca: {'K_1', 'K_2'}
        k_1: (float) - anisotropy constant first order
        K_2: (float) - anisotropy constant second order
    * E_demag: {'N_s'}
        N_s (list | ndarray) - demagnetizing factors [x,y,z]
    * E_me: {'B_1', 'B_2', 'eps_xx', 'eps_yy', 'eps_zz', 'eps_xy', 'eps_xz', 'eps_yz'}
        B_1: (float) - magnetoelastic coupling constant
        B_2: (float) - magnetoelastic coupling constant
        eps: (float) - deformations (by default, each component is 0)
    """

    if (E_ext is None) and (E_ca is None) and (E_demag is None) and (E_me is None):
        raise AttributeError("Должно быть хотя бы одно выражение для энергии.")
    if h is None:
        h = (stop - start) / 1000

    # Коэффициенты
    g = 1.919 * 10 ** 7  # gyromagnetic ratio of electron (gamma)
    if E_ext is not None:
        H = E_ext['H']
    if E_ca is not None:
        K_1 = E_ca['K_1']
        K_2 = E_ca['K_2']
    if E_demag is not None:
        N_s = np.array(E_demag['N_s'])
    if E_me is not None:
        B_1 = E_me['B_1']
        B_2 = E_me['B_2']
        eps_xx = E_me['eps_xx'] if 'eps_xx' in E_me else 0
        eps_yy = E_me['eps_yy'] if 'eps_yy' in E_me else 0
        eps_zz = E_me['eps_zz'] if 'eps_zz' in E_me else 0
        eps_xy = E_me['eps_xy'] if 'eps_xy' in E_me else 0
        eps_xz = E_me['eps_xz'] if 'eps_xz' in E_me else 0
        eps_yz = E_me['eps_yz'] if 'eps_yz' in E_me else 0

    def energy(m):
        """
        Подсчет полной энергии
        """
        result = 0
        m_sqd = m ** 2
        mx_sqd, my_sqd, mz_sqd = m_sqd[0], m_sqd[1], m_sqd[2]  # Компоненты вектора m в квадрате

        if E_ext is not None:  # Энергия от внешнего поля
            result -= M_s * np.dot(m, H)

        if E_ca is not None:  # Энергия кубической анизотропии
            result += K_1 * (mx_sqd * my_sqd + my_sqd * mz_sqd + mx_sqd * mz_sqd) + K_2 * (mx_sqd * my_sqd * mz_sqd)

        if E_demag is not None:  # Энергия размагничивания
            result += 0.5 * M_s ** 2 * np.dot(N_s, m_sqd)

        if E_me is not None:  # Магнитоупругая энергия
            result += B_1 * (mx_sqd * eps_xx + my_sqd * eps_yy + mz_sqd * eps_zz) + \
                      2 * B_2 * (m[0] * m[1] * eps_xy + m[0] * m[2] * eps_xz + m[1] * m[2] * eps_yz)

        return result

    def llg_equation(m: np.ndarray):
        m = m / np.linalg.norm(m)  # доп. нормировка на входе (bug fixed)
        """
        Уравнение LLG
        :param m: вектор намагниченности
        """
        h_M_s = h * M_s
        _energy = energy(m)
        H_eff_x = -(energy(m + [h, 0, 0]) - _energy) / h_M_s  # частные производные: dE/dm_x, dE/dm_y, dE/dm_z
        H_eff_y = -(energy(m + [0, h, 0]) - _energy) / h_M_s  # dE/dM = -1/M_s * dE/dm, M=M_s*m
        H_eff_z = -(energy(m + [0, 0, h]) - _energy) / h_M_s
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


def antiferromag(init_cond1: list | np.ndarray, init_cond2: list | np.ndarray, start, stop,
                 a: float, M_s: float, l: float,
                 E_ext: dict = None, E_ca: dict = None, E_demag: dict = None, E_me: dict = None, h=None):
    """
    Решение уравнения LLG методом Рунге-Кутты 4 порядка для ферромагнетика
    :param init_cond1: начальные условия для m1
    :param init_cond2: начальные условия для m2
    :param start: начало отрезка интегрирования (начальное время)
    :param stop: конец отрезка интегрирования (конечное время)
    :param a: damping coefficient (from 0 to 1)
    :param M_s: saturation magnetization
    :param l: exchange interaction field coefficient
    :param E_ext: энергия внешнего поля
    :param E_ca: энергия кубической анизотропии
    :param E_demag: энергия размагничивания
    :param E_me: магнитоупругая энергия
    :param h: шаг решения (по умолчанию: (stop - start) / 1000)
    :return: (t_val, m_val)
    t_val - значения времени
    m_val - значения вектора m (матрица размера n x 3; первый столбец - компоненты x, второй - y, третий - z)
    m_val = [[m_x0, m_y0, m_z0],[m_x1, m_y1, m_z1], ..., [m_xn, m_yn, m_zn]]

    ---coefficients for energies---
    * E_ext: {'H'}
        H: (list | ndarray) - внешнее поле
    * E_ca: {'K_1', 'K_2'}
        k_1: (float) - anisotropy constant first order
        K_2: (float) - anisotropy constant second order
    * E_demag: {'N_s'}
        N_s (list | ndarray) - demagnetizing factors [x,y,z]
    * E_me: {'B_1', 'B_2', 'eps_xx', 'eps_yy', 'eps_zz', 'eps_xy', 'eps_xz', 'eps_yz'}
        B_1: (float) - magnetoelastic coupling constant
        B_2: (float) - magnetoelastic coupling constant
        eps: (float) - deformations (by default, each component is 0)
    """
    if (E_ext is None) and (E_ca is None) and (E_demag is None) and (E_me is None):
        raise AttributeError("Должно быть хотя бы одно выражение для энергии.")
    if h is None:
        h = (stop - start) / 1000

    # Коэффициенты
    g = 1.919 * 10 ** 7  # gyromagnetic ratio of electron (gamma)
    if E_ext is not None:
        H = E_ext['H']
    if E_ca is not None:
        K_0 = E_ca['K_0']
        K_1 = E_ca['K_1']
    if E_demag is not None:
        N_s = np.array(E_demag['N_s'])
    if E_me is not None:
        B_1 = E_me['B_1']
        B_2 = E_me['B_2']
        eps_xx = E_me['eps_xx'] if 'eps_xx' in E_me else 0
        eps_yy = E_me['eps_yy'] if 'eps_yy' in E_me else 0
        eps_zz = E_me['eps_zz'] if 'eps_zz' in E_me else 0
        eps_xy = E_me['eps_xy'] if 'eps_xy' in E_me else 0
        eps_xz = E_me['eps_xz'] if 'eps_xz' in E_me else 0
        eps_yz = E_me['eps_yz'] if 'eps_yz' in E_me else 0

    def energy(m1, m2):
        """
        Подсчет полной энергии
        """
        result = 0
        neel_vector = (m1 - m2) / 2

        if E_ext is not None:  # Энергия от внешнего поля
            result -= M_s * np.dot(neel_vector, H)

        if E_ca is not None:  # Энергия анизотропии
            result += K_0 * neel_vector[2] ** 2 + K_1 * neel_vector[1] ** 2

        if E_demag is not None:  # Энергия размагничивания
            result += 0.5 * M_s ** 2 * np.dot(N_s, neel_vector ** 2)

        return result

    def llg_equation(m1: np.ndarray, m2: np.ndarray):
        """
        Уравнение LLG
        :param m1: вектор намагниченности
        :param m2: вектор намагниченности
        """
        m1 = m1 / np.linalg.norm(m1)  # доп. нормировка на входе (bug fixed)
        m2 = m2 / np.linalg.norm(m2)
        h_M_s = h * M_s
        _energy = energy(m1, m2)

        H1_eff_x = -(energy(m1 + [h, 0, 0], m2) - _energy) / h_M_s  # частные производные: dE/dm_x, dE/dm_y, dE/dm_z
        H1_eff_y = -(energy(m1 + [0, h, 0], m2) - _energy) / h_M_s  # dE/dM = -1/M_s * dE/dm, M=M_s*m
        H1_eff_z = -(energy(m1 + [0, 0, h], m2) - _energy) / h_M_s
        H1_eff = np.array([H1_eff_x, H1_eff_y, H1_eff_z]) - l * m2  # полное магнитное поле для m1

        H2_eff_x = -(energy(m1, m2 + [h, 0, 0]) - _energy) / h_M_s
        H2_eff_y = -(energy(m1, m2 + [0, h, 0]) - _energy) / h_M_s
        H2_eff_z = -(energy(m1, m2 + [0, 0, h]) - _energy) / h_M_s
        H2_eff = np.array([H2_eff_x, H2_eff_y, H2_eff_z]) - l * m1  # полное магнитное поле для m2

        def llg(m, H_eff):
            mxH = np.cross(m, H_eff)  # первое слагаемое в скобках уравнения (m x H_eff)
            mx_mxH = np.cross(m, mxH)  # второе слагаемое (m x (m x H_eff))
            return (- g / (1 + a ** 2)) * (mxH + a * mx_mxH)

        return np.array([llg(m1, H1_eff), llg(m2, H2_eff)])

    n = int((stop - start) / h)  # количество разбиений
    t_val = np.zeros(n)
    t_val[0] = start
    m_val = np.zeros([2, n, 3])  # матрица векторов m
    init_cond1 = init_cond1 / np.linalg.norm(init_cond1)  # нормализация начальных условий
    init_cond2 = init_cond2 / np.linalg.norm(init_cond2)  # нормализация начальных условий
    m_val[0][0], m_val[1][0] = init_cond1, init_cond2

    for i in range(1, n):
        t = t_val[i - 1]
        m1, m2 = m_val[0][i - 1], m_val[1][i - 1]

        # Коэффициенты Рунге-Кутты
        k1_1, k2_1 = h * llg_equation(m1, m2)
        k1_2, k2_2 = h * llg_equation(m1 + k1_1 / 2, m2 + k2_1 / 2)
        k1_3, k2_3 = h * llg_equation(m1 + k1_2 / 2, m2 + k2_2 / 2)
        k1_4, k2_4 = h * llg_equation(m1 + k1_3, m2 + k2_3)

        m1_new = m1 + (k1_1 + 2 * k1_2 + 2 * k1_3 + k1_4) / 6
        m1_new = m1_new / np.linalg.norm(m1_new)  # нормализация |m|=1

        m2_new = m2 + (k2_1 + 2 * k2_2 + 2 * k2_3 + k2_4) / 6
        m2_new = m2_new / np.linalg.norm(m2_new)

        t_val[i] = t + h
        m_val[0][i] = m1_new
        m_val[1][i] = m2_new

    return t_val, m_val


if __name__ == '__main__':
    m_0 = [0, 1, 1]
    m_0 = m_0 / np.linalg.norm(m_0)

    H = [3, 1, 2]
    E_ext = {'H': H}
    E_ca = {'K_1': 4.2 * 10 ** 5, 'K_2': 1.5 * 10 ** 5}
    E_demag = {'N_s': [0, 0, 4 * np.pi]}
    E_me = {'B_1': -2.9 * 10 ** 7, 'B_2': 3.2 * 10 ** 7, 'eps_xx': -0.63, 'eps_yy': -0.63}

    t, m = ferromag(m_0, 0, 10 ** (-7), a=0.1, M_s=1707, E_ca=E_ca)
    plt.plot(t, m, label=['m_x', 'm_y', 'm_z'])
    m_0 = list(m_0)
    m_0[0], m_0[1], m_0[2] = round(m_0[0], 10), round(m_0[1], 10), round(m_0[2], 10)
    plt.title(f'{H=} {m_0=}')

    plt.legend()
    plt.grid()
    plt.savefig('data_llg/check2.png')
    plt.show()
