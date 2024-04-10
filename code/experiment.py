import numpy as np
import pandas as pd
from dataclasses import dataclass

from synthesizer import Synthesizer
from generator import Generator
from algorithm import Algorithm

from hypers import weight_hypers
from hypers import alpha_hypers

# Fixed params

series_type = "default"
from_start = False
a, b = -40., 40.
seeds = 5 * np.arange(2, 6) + 1

dim = 10
low, high = -10, 10
clip = (a, b)
noise_var = 1
workers_num = 3

length = 1500
lower_bound, upper_bound = 100, 400
alternating=True


@dataclass
class Params:
    series_type: str
    from_start: bool
    a: float
    b: float

    dim: int
    low: int
    high: int
    clip: tuple[float, float]
    noise_var: float
    workers_num: int

    length: int
    lower_bound: int
    upper_bound: int
    alternating: bool


@dataclass
class Logs:
    experts_losses_all: np.array
    master_losses_all: np.array
    weights_all: np.array
    ideal_losses: np.array
    theoretical_upper: np.array


@dataclass
class Experiment:
    random_seed: int

    key_w: str
    key_a: str
    train_window: int

    params: Params
    logs: Logs
    regret: float


def run_experiments(windows, interesting_w, interesting_a):
    interesting = set()

    for key_w in interesting_w:
        interesting.add((key_w, "default"))

    for key_a in interesting_a:
        interesting.add(("simple_101", key_a))

    experiments = []

    for rs in seeds:
        synt = Synthesizer(series_type, dim=dim, low=low, high=high, clip=clip,
                           noise_var=noise_var, workers_num=workers_num, random_seed=rs)
        gen = Generator(series_type, synt)
        gen.generate(length=length, from_start=from_start,
                     lower_bound=lower_bound, upper_bound=upper_bound, alternating=True)

        for train_window in windows:
            for key_w, key_a in interesting:
                hyper_w = weight_hypers[key_w]
                hyper_a = alpha_hypers[key_a]
                algo = Algorithm(series_type, gen, train_window=train_window, a=a, b=b,
                                 weights_func=hyper_w.func, weight_const=hyper_w.const, alpha_func=hyper_a.func,
                                 init_pretrained=False)
                gen.launch()
                algo.run()
                algo.post_calculations(from_start=from_start)

                params = Params(series_type=series_type,
                                from_start=from_start,
                                a=a,
                                b=b,
                                dim=dim,
                                low=low,
                                high=high,
                                clip=clip,
                                noise_var=noise_var,
                                workers_num=workers_num,
                                length=length,
                                lower_bound=lower_bound,
                                upper_bound=upper_bound,
                                alternating=alternating)

                logs = Logs(experts_losses_all=algo.experts_losses_all,
                            master_losses_all=algo.master_losses_all,
                            weights_all=algo.weights_all,
                            ideal_losses=algo.ideal_losses,
                            theoretical_upper=algo.theoretical_upper)

                experiment = Experiment(random_seed=rs,
                                        key_w=key_w,
                                        key_a=key_a,
                                        train_window=train_window,
                                        params=params,
                                        logs=logs,
                                        regret=algo.regret)

                experiments.append(experiment)

                del algo

    dct = {}
    for experiment in experiments:
        key = (experiment.train_window, experiment.key_w, experiment.key_a)
        if key not in dct:
            hyper_w = weight_hypers[experiment.key_w]
            hyper_a = alpha_hypers[experiment.key_a]
            dct[key] = [experiment.train_window, hyper_w.repr, hyper_a.repr]
        dct[key].append(round(experiment.regret, 2))

    df = pd.DataFrame(dct).transpose()

    labels = ["train_window", "weight_function", "alpha_function"]

    renaming={}
    for i in list(df):
        if i < 3:
            renaming[i] = labels[i]
        else:
            renaming[i] = f"random_{i-2}"

    df = df.set_index(np.arange(df.index.size)).rename(columns=renaming)
    ordered_repr = windows + [wh.repr for wh in weight_hypers.values()] + [ah.repr for ah in alpha_hypers.values()]
    df = df.sort_values(labels, key=lambda col: col.apply(lambda x: ordered_repr.index(x)))
    df.insert(3, "mean", df.iloc[:, 3:].mean(axis=1))
    df["mean"] = df["mean"].astype(float).round(2)

    return experiments, df










