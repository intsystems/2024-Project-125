import numpy as np
import pandas as pd
from dataclasses import dataclass

from synthesizer import Synthesizer
from generator import Generator
from algorithm import Algorithm

from hypers import weight_hypers
from hypers import alpha_hypers


@dataclass
class Result:
    seed: int
    key_w: str
    key_a: str
    train_window: int

    algo: Algorithm
    regret: float


series_type = "default"
from_start=False
a, b = -40, 40

seeds = 5 * np.arange(2, 6) + 1
windows = [5, 10, 20]

interesting = set()

interesting_w = ["default", "simple_105", "simple_11", "simple_12", "simple_2"]
interesting_a = ["default", "simple_11", "simple_13", "simple_15", "simple_2"]

for key_w in interesting_w:
    interesting.add((key_w, "default"))

for key_a in interesting_a:
    interesting.add(("simple_105", key_a))

results = []

for rs in seeds:
    synt = Synthesizer(series_type, dim=10, low=-10, high=10, clip=(a, b),
                       noise_var=1, workers_num=3, random_seed=rs)
    gen = Generator(series_type, synt)
    gen.generate(length=1500, from_start=from_start, lower_bound=100, upper_bound=400, alternating=True)

    for train_window in windows:
        algos_w = []
        for key_w, key_a in interesting:
            hyper_w = weight_hypers[key_w]
            hyper_a = alpha_hypers[key_a]
            algo = Algorithm(series_type, gen, train_window=train_window, a=a, b=b,
                             weights_func=hyper_w.func, weight_const=hyper_w.const, alpha_func=hyper_a.func,
                             init_pretrained=False)
            gen.launch()
            algo.run()
            algo.post_calculations(from_start=from_start)

            result = Result(rs, key_w, key_a, train_window, algo, algo.regret)
            results.append(result)









