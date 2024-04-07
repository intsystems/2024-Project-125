import numpy as np
from dataclasses import dataclass


DEFAULT_CONST = 2.10974  # paper


def default_weights_func(x):
    return 1 / ((x + 1) * np.square(np.log(x + 1))) / DEFAULT_CONST


CONST_105 = 20.5808  # wolfram alpha


def weights_func_105(x):
    return 1 / (x ** 1.05) / CONST_105


CONST_11 = 10.5844  # wolfram alpha


def weights_func_11(x):
    return 1 / (x ** 1.1) / CONST_11


CONST_115 = 7.25469  # wolfram alpha


def weights_func_115(x):
    return 1 / (x ** 1.15) / CONST_115


CONST_12 = 5.59158  # wolfram alpha


def weights_func_12(x):
    return 1 / (x ** 1.2) / CONST_12


CONST_14 = 3.10555  # wolfram alpha


def weights_func_14(x):
    return 1 / (x ** 1.4) / CONST_14


CONST_17 = 2.05429  # wolfram alpha


def weights_func_17(x):
    return 1 / (x ** 1.7) / CONST_17


CONST_2 = np.pi ** 2 / 6


def weights_func_2(x):
    return 1 / (x ** 2) / CONST_2


CONST_3 = 1.2021  # wolfram alpha


def weights_func_3(x):
    return 1 / (x ** 3) / CONST_3


CONST_SLOW = 0. # wolfram does not know


def weights_func_slow(x):
    return 1 / ((x + 1) * np.log(x + 1) * np.square(np.log(np.log(x + 1)))) / CONST_SLOW


@dataclass
class WeightsHyper:
    const: float
    func: callable(float)
    repr: str


weight_hypers = {
    "default": WeightsHyper(DEFAULT_CONST, default_weights_func, "1 / ((x + 1) * (ln(x + 1))^2)"),
    "simple_105": WeightsHyper(CONST_105, weights_func_105, "1 / (x^1.05)"),
    "simple_11": WeightsHyper(CONST_11, weights_func_11, "1 / (x^1.1)"),
    "simple_115": WeightsHyper(CONST_115, weights_func_115, "1 / (x^1.15)"),
    "simple_12": WeightsHyper(CONST_12, weights_func_12, "1 / (x^1.2)"),
    "simple_14": WeightsHyper(CONST_14, weights_func_14, "1 / (x^1.4)"),
    "simple_17": WeightsHyper(CONST_17, weights_func_17, "1 / (x^1.7)"),
    "simple_2": WeightsHyper(CONST_2, weights_func_2, "1 / (x^2)"),
    "simple_3": WeightsHyper(CONST_3, weights_func_3, "1 / (x^3)"),
    "inv_exp": WeightsHyper(CONST_SLOW, weights_func_slow, "1 / ((x + 1) * np.log(x + 1) * np.square(np.log(np.log(x + 1))))"),
}


# ------------------------------------------------


def default_alpha_func(x):
    return 1 / (x + 1)


def alpha_func_11(x):
    return 1 / (x + 1) ** 1.1


def alpha_func_12(x):
    return 1 / (x + 1) ** 1.2


def alpha_func_13(x):
    return 1 / (x + 1) ** 1.3


def alpha_func_14(x):
    return 1 / (x + 1) ** 1.4


def alpha_func_15(x):
    return 1 / (x + 1) ** 1.5


def alpha_func_16(x):
    return 1 / (x + 1) ** 1.6


def alpha_func_17(x):
    return 1 / (x + 1) ** 1.7


def alpha_func_2(x):
    return 1 / (x + 1) ** 2


def alpha_func_3(x):
    return 1 / (x + 1) ** 3


@dataclass
class AlphaHyper:
    func: callable(float)
    repr: str


alpha_hypers = {
    "default": AlphaHyper(default_alpha_func, "1 / (x + 1)"),
    "simple_11": AlphaHyper(alpha_func_11, "1 / (x + 1)^1.1"),
    "simple_12": AlphaHyper(alpha_func_12, "1 / (x + 1)^1.2"),
    "simple_13": AlphaHyper(alpha_func_13, "1 / (x + 1)^1.3"),
    "simple_14": AlphaHyper(alpha_func_14, "1 / (x + 1)^1.4"),
    "simple_15": AlphaHyper(alpha_func_15, "1 / (x + 1)^1.5"),
    "simple_16": AlphaHyper(alpha_func_16, "1 / (x + 1)^1.6"),
    "simple_17": AlphaHyper(alpha_func_17, "1 / (x + 1)^1.7"),
    "simple_2": AlphaHyper(alpha_func_2, "1 / (x + 1)^2)"),
    "simple_3": AlphaHyper(alpha_func_3, "1 / (x + 1)^3"),
}
