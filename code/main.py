from experiment import run_experiments

windows = [10, 50, 100]
interesting_w = ["default", "simple_101", "exp_4"]
interesting_a = ["default", "log", "exp"]

experiments, df = run_experiments(windows, interesting_w, interesting_a)
