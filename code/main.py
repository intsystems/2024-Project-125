from experiment import run_experiments

windows = [5, 10, 20, 100]
# noises = [0.0001, 0.001, 0.1, 1, 10, 100]
different_wf = ["default", "simple_101", "simple_11", "simple_2"]
different_af = ["default", "simple_05", "simple_2", "exp"]
filepath = '../results/experiment_main.json'
experiments, df = run_experiments(windows, different_wf, different_af, filepath)
