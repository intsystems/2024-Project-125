from experiment import run_experiments

windows = [5, 10, 20]
# noises = [0.0001, 0.001, 0.1, 1, 10, 100]
interesting_w = []
interesting_a = ["default", "simple_05"]
filepath = '../results/experiment_1.json'
experiments, df = run_experiments(windows, interesting_w, interesting_a, filepath)
