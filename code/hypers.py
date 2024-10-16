import numpy as np
from dataclasses import dataclass


DEFAULT_CONST = 2.10974


def default_weights_func(x):
    return 1 / ((x + 1) * np.square(np.log(x + 1))) / DEFAULT_CONST


CONST_1001 = 1000.58


def weights_func_1001(x):
    return 1 / (x ** 1.001) / CONST_1001


CONST_101 = 100.578


def weights_func_101(x):
    return 1 / (x ** 1.01) / CONST_101


CONST_105 = 20.5808


def weights_func_105(x):
    return 1 / (x ** 1.05) / CONST_105


CONST_11 = 10.5844


def weights_func_11(x):
    return 1 / (x ** 1.1) / CONST_11


CONST_12 = 5.59158


def weights_func_12(x):
    return 1 / (x ** 1.2) / CONST_12


CONST_2 = np.pi ** 2 / 6


def weights_func_2(x):
    return 1 / (x ** 2) / CONST_2


CONST_3 = 1.2021


def weights_func_3(x):
    return 1 / (x ** 3) / CONST_3


# CONST_EXP_4 = 1 / (np.exp(1/4) - 1)  // little errors
CONST_EXP_4 = 3.520812


def weights_exp4_func(x):
    return np.exp(-x / 4) / CONST_EXP_4


CONST_SLOW = 2.3533


def weights_func_slow(x):
    return 1 / ((x + 4) * np.log(x + 4) * np.square(np.log(np.log(x + 4)))) / CONST_SLOW


def weights_func_slow_1(x):
    return 1 / ((x + 1) * np.log(x + 1) * np.square(np.log(np.log(x + 4))))


def weights_func_slow_20(x):
    return 1 / ((x + 20) * np.log(x + 20) * np.square(np.log(np.log(x + 20))))


def weights_func_slow_100(x):
    return 1 / ((x + 100) * np.log(x + 100) * np.square(np.log(np.log(x + 100))))


CONST = 4000.


def weights_func_const(x):
    return (1 - 0.00001 * x) / CONST


def weights_func_diverge_1(x):
    return 1 / x / CONST


def weights_func_diverge_01(x):
    return 1 / (x ** 0.1) / CONST


def weights_func_diverge_03(x):
    return 1 / (x ** 0.3) / CONST


def weights_func_diverge_05(x):
    return 1 / (x ** 0.5) / CONST


def weights_func_diverge_07(x):
    return 1 / (x ** 0.7) / CONST


def weights_func_diverge_09(x):
    return 1 / (x ** 0.9) / CONST


def weights_func_shift_10(x):
    return 1 / (x + 10) / CONST


def weights_func_shift_100(x):
    return 1 / (x + 100) / CONST


def weights_func_shift_1000(x):
    return 1 / (x + 1000) / CONST


CONST_D_1 = 1.

CONST_D_10 = 10.

CONST_D_100 = 100.

CONST_D_1000 = 1000.

CONST_D_10000 = 10000.


@dataclass
class WeightsHyper:
    const: float
    func: callable(float)
    repr: str


weight_hypers = {
    "default": WeightsHyper(DEFAULT_CONST, default_weights_func, "1 / ((x + 1) * (ln(x + 1))^2)"),
    "simple_1001": WeightsHyper(CONST_1001, weights_func_1001, "1 / (x^1.001)"),
    "simple_101": WeightsHyper(CONST_101, weights_func_101, "1 / (x^1.01)"),
    "simple_105": WeightsHyper(CONST_105, weights_func_105, "1 / (x^1.05)"),
    "simple_11": WeightsHyper(CONST_11, weights_func_11, "1 / (x^1.1)"),
    "simple_12": WeightsHyper(CONST_12, weights_func_12, "1 / (x^1.2)"),
    "simple_2": WeightsHyper(CONST_2, weights_func_2, "1 / (x^2)"),
    "simple_3": WeightsHyper(CONST_3, weights_func_3, "1 / (x^3)"),
    "exp_4": WeightsHyper(CONST_EXP_4, weights_exp4_func, "1 / e^(x/4)"),
    "slow": WeightsHyper(CONST_SLOW, weights_func_slow, "1 / ((x + 4) * ln(x + 4) * (ln(ln(x + 4)))^2"),
    "slow_1": WeightsHyper(CONST_SLOW, weights_func_slow_1, "1 / ((x + 1) * ln(x + 1) * (ln(ln(x + 4)))^2"),
    "slow_20": WeightsHyper(CONST_SLOW, weights_func_slow_20, "1 / ((x + 20) * ln(x + 20) * (ln(ln(x + 20)))^2"),
    "slow_100": WeightsHyper(CONST_SLOW, weights_func_slow_100, "1 / ((x + 100) * ln(x + 100) * (ln(ln(x + 100)))^2"),
    "const": WeightsHyper(CONST, weights_func_const, "1 / c"),
    "diverge_1": WeightsHyper(CONST, weights_func_diverge_1, "1 / x"),
    "diverge_01": WeightsHyper(CONST, weights_func_diverge_01, "1 / (x^0.1)"),
    "diverge_03": WeightsHyper(CONST, weights_func_diverge_03, "1 / (x^0.3)"),
    "diverge_05": WeightsHyper(CONST, weights_func_diverge_05, "1 / (x^0.5)"),
    "diverge_07": WeightsHyper(CONST, weights_func_diverge_07, "1 / (x^0.7)"),
    "diverge_09": WeightsHyper(CONST, weights_func_diverge_09, "1 / (x^0.9)"),
    "shift_10": WeightsHyper(CONST, weights_func_shift_10, "1 / (x + 10)"),
    "shift_100": WeightsHyper(CONST, weights_func_shift_100, "1 / (x + 100)"),
    "shift_1000": WeightsHyper(CONST, weights_func_shift_1000, "1 / (x + 1000)"),

    "default_d1": WeightsHyper(CONST_D_1, default_weights_func, "1 / ((x + 1) * (ln(x + 1))^2)d1"),
    "default_d100": WeightsHyper(CONST_D_100, default_weights_func, "1 / ((x + 1) * (ln(x + 1))^2)d100"),

    "const_d1": WeightsHyper(CONST_D_1, weights_func_const, "1 / d1"),
    "const_d100": WeightsHyper(CONST_D_100, weights_func_const, "1 / d100"),

    "diverge_05_d1": WeightsHyper(CONST_D_1, weights_func_diverge_05, "1 / (x^0.5)d1"),
    "diverge_05_d100": WeightsHyper(CONST_D_100, weights_func_diverge_05, "1 / (x^0.5)d100"),

    "simple_101_d1": WeightsHyper(CONST_D_1, weights_func_101, "1 / (x^1.01)d1"),
    "simple_101_d100": WeightsHyper(CONST_D_100, weights_func_101, "1 / (x^1.01)d100"),
}


# ------------------------------------------------


def alpha_func_const_1100(x):
    return 1 / 100


def alpha_func_const_1500(x):
    return 1 / 500


def alpha_func_const_11000(x):
    return 1 / 1000


def alpha_func_const_15000(x):
    return 1 / 5000


def alpha_func_const_110000(x):
    return 1 / 10000


def alpha_func_const_150000(x):
    return 1 / 50000


def default_alpha_func(x):
    return 1 / (x + 1)


def alpha_func_01(x):
    return 1 / (x + 1) ** 0.1


def alpha_func_05(x):
    return 1 / (x + 1) ** 0.5


def alpha_func_11(x):
    return 1 / (x + 1) ** 1.1


def alpha_func_12(x):
    return 1 / (x + 1) ** 1.2


def alpha_func_15(x):
    return 1 / (x + 1) ** 1.5


def alpha_func_2(x):
    return 1 / (x + 1) ** 2


def alpha_func_3(x):
    return 1 / (x + 1) ** 3


def alpha_func_shift_10(x):
    return 1 / (x + 10)


def alpha_func_shift_100(x):
    return 1 / (x + 100)


def alpha_func_shift_1000(x):
    return 1 / (x + 1000)


def alpha_crook_func(x):
    return 1 / ((x + 1) * np.square(np.log(x + 1)))


def alpha_log_func(x):
    return 1 / np.log(x + 1)


def alpha_exp_func(x):
    return np.exp(-x)


def alpha_exp_3_func(x):
    return np.exp(-x / 3)


@dataclass
class AlphaHyper:
    func: callable(float)
    repr: str


alpha_hypers = {
    "default": AlphaHyper(default_alpha_func, "1 / (x + 1)"),
    "const_1100": AlphaHyper(alpha_func_const_1100, "1 / 100"),
    "const_1500": AlphaHyper(alpha_func_const_1500, "1 / 500"),
    "const_11000": AlphaHyper(alpha_func_const_11000, "1 / 1000"),
    "const_15000": AlphaHyper(alpha_func_const_15000, "1 / 5000"),
    "const_110000": AlphaHyper(alpha_func_const_110000, "1 / 10000"),
    "const_150000": AlphaHyper(alpha_func_const_150000, "1 / 50000"),
    "simple_01": AlphaHyper(alpha_func_01, "1 / (x + 1)^0.1"),
    "simple_05": AlphaHyper(alpha_func_05, "1 / (x + 1)^0.5"),
    "simple_11": AlphaHyper(alpha_func_11, "1 / (x + 1)^1.1"),
    "simple_12": AlphaHyper(alpha_func_12, "1 / (x + 1)^1.2"),
    "simple_15": AlphaHyper(alpha_func_15, "1 / (x + 1)^1.5"),
    "simple_2": AlphaHyper(alpha_func_2, "1 / (x + 1)^2"),
    "simple_3": AlphaHyper(alpha_func_3, "1 / (x + 1)^3"),
    "shift_10": AlphaHyper(alpha_func_shift_10, "1 / (x + 10)"),
    "shift_100": AlphaHyper(alpha_func_shift_100, "1 / (x + 100)"),
    "shift_1000": AlphaHyper(alpha_func_shift_1000, "1 / (x + 1000)"),
    "crook": AlphaHyper(alpha_crook_func, "1 / ((x + 1)*(ln(x + 1))^2)"),
    "log": AlphaHyper(alpha_log_func, "1 / ln(x + 1)"),
    "exp": AlphaHyper(alpha_exp_func, "1 / e^x"),
    "exp_3": AlphaHyper(alpha_exp_3_func, "1 / e^(x/3)"),
}
