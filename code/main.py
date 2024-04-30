from experiment import run_experiments
from experiment import Params
import numpy as np

filepath = 'results/experiment_main.json'

seeds = 5 * np.arange(2, 5) + 111

params = Params(series_type="default",
                from_start=False,
                a=-40,
                b=40,
                dim=10,
                low=-10,
                high=10,
                clip=(-40, 40),
                workers_num=3,
                length=2000,
                lower_bound=100,
                upper_bound=400,
                alternating=True)

different_noises = [5, 10, 20, 50, 100, 200]
different_windows = [0.0001, 0.001, 0.1, 1, 10, 100]
different_wf = ["default", "simple_101", "simple_2", "diverge_05", "const"]
different_af = ["default", "simple_05", "simple_2", "exp"]

interesting = set()

for key_w in different_wf:
    interesting.add((key_w, "default"))

for key_a in different_af:
    interesting.add(("default", key_a))

experiments, df = run_experiments(filepath, seeds, params, different_noises, different_windows, interesting)
