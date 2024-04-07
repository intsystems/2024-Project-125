import numpy as np
import pandas as pd

from synthesizer import Synthesizer
from generator import Generator
from algorithm import Algorithm

from hypers import weight_hypers
from hypers import alpha_hypers

series_type = "default"
from_start=False
a, b = -40, 40
train_window = 10

generators = []

for rs in 5 * np.arange(2, 6) + 1:
    synt = Synthesizer(series_type, dim=10, low=-10, high=10, clip=(a, b),
                       noise_var=1, workers_num=3, random_seed=rs)
    gen = Generator(series_type, synt)
    gen.generate(length=1500, from_start=from_start, lower_bound=100, upper_bound=400, alternating=True)
    generators.append(gen)


    synt = Synthesizer(series_type, dim=20, low=-10, high=10, clip=(2 * a, 2 * b),
                       noise_var=1, workers_num=3, random_seed=rs)
    gen = Generator(series_type, synt)
    gen.generate(length=1500, from_start=from_start, lower_bound=100, upper_bound=400, alternating=True)
    generators.append(gen)


    synt = Synthesizer(series_type, dim=40, low=-10, high=10, clip=(4 * a, 4 * b),
                       noise_var=1, workers_num=3, random_seed=rs)
    gen = Generator(series_type, synt)
    gen.generate(length=1500, from_start=from_start, lower_bound=100, upper_bound=400, alternating=True)
    generators.append(gen)



interesting_w = ["default", "simple_105", "simple_11", "simple_12", "simple_2"]
interesting_a = ["default", "simple_11", "simple_13", "simple_15", "simple_2"]

dict_w = {}
dict_a = {}

for key in interesting_w:
    w_hyper = weight_hypers[key]
    dict_w[key] = [w_hyper.repr]

for key in interesting_a:
    a_hyper = alpha_hypers[key]
    dict_a[key] = [a_hyper.repr]

for gen in generators:
    algos_w = []
    for key in interesting_w:
        w_hyper = weight_hypers[key]
        algo = Algorithm(series_type, gen, train_window=train_window, a=a, b=b,
                         weights_func=w_hyper.func, weight_const=w_hyper.const,
                         init_pretrained=False)
        gen.launch()
        algo.run()
        algo.post_calculations(from_start=from_start)
        algos_w.append(algo)
        dict_w[key].append(algo.regret)

    best = weight_hypers["simple_105"]
    algos_a = []

    for key in interesting_a:
        a_hyper = alpha_hypers[key]
        algo = Algorithm(series_type, gen, train_window=train_window, a=a, b=b,
                         weights_func=best.func, weight_const=best.const,
                         alpha_func=a_hyper.func,
                         init_pretrained=False)
        gen.launch()
        algo.run()
        algo.post_calculations(from_start=from_start)
        algos_a.append(algo)
        dict_a[key].append(algo.regret)


df_w = pd.DataFrame(dict_w).transpose().rename(columns={0 : "weights_function"})
df_a = pd.DataFrame(dict_a).transpose().rename(columns={0 : "alpha_function"})






