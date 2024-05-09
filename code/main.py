from experiment import run_experiments
from experiment import load_experiments
from experiment import Params
import numpy as np

filepath = 'results/experiment_main.json'

seeds = np.arange(4) + 101

params = Params(series_type="default",
                from_start=False,
                a=-40,
                b=40,
                dim=10,
                low=-10,
                high=10,
                clip=(-40, 40),
                workers_num=5,
                length=2000,
                lower_bound=50,
                upper_bound=300,
                alternating=True)

different_noises = [1, 10]
different_windows = [10, 20]

different_wf = ["simple_101", "const", "diverge_05"]
different_af = ["default", "simple_05", "simple_2", "shift_100"]
mixing_types = ["start", "uniform past", "decaying past"]

interesting = set()

for mixing_type in mixing_types:
    for key_a in different_af:
        for key_w in different_wf:
            interesting.add((key_w, mixing_type, key_a))

experiments, df = run_experiments(filepath, seeds, params, different_noises, different_windows, interesting)

# experiments, df = load_experiments(filepath)
