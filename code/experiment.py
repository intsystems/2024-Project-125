import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import dataclasses_json
import dataclasses

from synthesizer import Synthesizer
from generator import Generator
from algorithm import Algorithm

from hypers import weight_hypers
from hypers import alpha_hypers


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class Params:
    series_type: str
    from_start: bool
    a: float
    b: float

    dim: int
    low: int
    high: int
    clip: tuple[float, float]
    workers_num: int

    length: int
    lower_bound: int
    upper_bound: int
    alternating: bool


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class Logs:
    total_time: int
    shift: int
    indexes: list[int]
    stamps: np.array
    master_losses_all: np.array
    ideal_losses: np.array
    theoretical_upper: np.array


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class Experiment:
    filepath: str
    random_seed: int

    key_w: str
    key_a: str
    train_window: int
    noise_var: float

    params: Params
    logs: Logs
    regret: float

    @classmethod
    def from_dictionary(cls, exp_dict, filepath):
        params = Params(
            series_type=exp_dict['params']['series_type'],
            from_start=exp_dict['params']['from_start'],
            a=exp_dict['params']['a'],
            b=exp_dict['params']['b'],
            dim=exp_dict['params']['dim'],
            low=exp_dict['params']['low'],
            high=exp_dict['params']['high'],
            clip=tuple(exp_dict['params']['clip']),
            workers_num=exp_dict['params']['workers_num'],
            length=exp_dict['params']['length'],
            lower_bound=exp_dict['params']['lower_bound'],
            upper_bound=exp_dict['params']['upper_bound'],
            alternating=exp_dict['params']['alternating'],
        )

        logs = Logs(
            total_time=exp_dict['logs']['total_time'],
            shift=exp_dict['logs']['shift'],
            indexes=exp_dict['logs']['indexes'],
            stamps=np.array(exp_dict['logs']['stamps']),
            master_losses_all=np.array(exp_dict['logs']['master_losses_all']),
            ideal_losses=np.array(exp_dict['logs']['ideal_losses']),
            theoretical_upper=np.array(exp_dict['logs']['theoretical_upper']),
        )

        return Experiment(
            filepath=filepath,
            random_seed=exp_dict['random_seed'],
            key_w=exp_dict['key_w'],
            key_a=exp_dict['key_a'],
            train_window=exp_dict['train_window'],
            noise_var=exp_dict['noise_var'],
            params=params,
            logs=logs,
            regret=exp_dict['regret'],
        )


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def run_experiments(filepath, seeds, params, different_noises, different_windows, different_wf, different_af):
    interesting = set()

    for key_w in different_wf:
        interesting.add((key_w, "default"))

    for key_a in different_af:
        interesting.add(("simple_101", key_a))

    experiments = []

    for rs in seeds:
        for noise_var in different_noises:
            for train_window in different_windows:
                synt = Synthesizer(params.series_type, dim=params.dim,
                                   low=params.low, high=params.high, clip=params.clip,
                                   noise_var=noise_var, workers_num=params.workers_num, random_seed=rs)
                gen = Generator(params.series_type, synt)
                gen.generate(length=params.length, from_start=params.from_start,
                             lower_bound=params.lower_bound, upper_bound=params.upper_bound,
                             alternating=params.alternating)

                for key_w, key_a in interesting:
                    hyper_w = weight_hypers[key_w]
                    hyper_a = alpha_hypers[key_a]
                    algo = Algorithm(params.series_type, gen, train_window=train_window, a=params.a, b=params.b,
                                     weights_func=hyper_w.func, weight_const=hyper_w.const, alpha_func=hyper_a.func,
                                     init_pretrained=False)
                    gen.launch()
                    algo.run()
                    algo.post_calculations(from_start=params.from_start)

                    logs = Logs(total_time=algo.total_time,
                                shift=algo.shift,
                                indexes=gen.indexes,
                                stamps=gen.stamps,
                                master_losses_all=algo.master_losses_all,
                                ideal_losses=algo.ideal_losses,
                                theoretical_upper=algo.theoretical_upper,
                                )

                    experiment = Experiment(filepath=filepath,
                                            random_seed=rs,
                                            key_w=key_w,
                                            key_a=key_a,
                                            train_window=train_window,
                                            noise_var=noise_var,
                                            params=params,
                                            logs=logs,
                                            regret=algo.regret
                                            )

                    experiments.append(experiment)

                    del algo
                del gen
                del synt

    df = experiments_to_df(experiments)

    if filepath is not None:
        # df.to_csv(filepath, sep='\t')
        with open(filepath, 'w') as f:
            json.dump(experiments, f, cls=EnhancedJSONEncoder)

    return experiments, df


def experiments_to_df(experiments):
    dct = {}
    for experiment in experiments:
        key = (experiment.noise_var, experiment.train_window, experiment.key_w, experiment.key_a)
        if key not in dct:
            hyper_w = weight_hypers[experiment.key_w]
            hyper_a = alpha_hypers[experiment.key_a]
            dct[key] = [experiment.noise_var, experiment.train_window, hyper_w.repr, hyper_a.repr]
        dct[key].append(round(experiment.regret, 2))

    df = pd.DataFrame(dct).transpose()

    labels = ["noise_var", "train_window", "weight_function", "alpha_function"]
    indent = len(labels)

    renaming = {}
    for i in list(df):
        if i < indent:
            renaming[i] = labels[i]
        else:
            renaming[i] = f"random_{i - indent}"

    df = df.set_index(np.arange(df.index.size)).rename(columns=renaming)
    ordered_repr = (sorted(df['noise_var'].unique()) +
                    sorted(df['train_window'].unique()) +
                    [wh.repr for wh in weight_hypers.values()] +
                    [ah.repr for ah in alpha_hypers.values()])
    df = df.sort_values(labels, key=lambda col: col.apply(lambda x: ordered_repr.index(x)))
    df.insert(indent, "regret", df.iloc[:, indent:].mean(axis=1))
    df["regret"] = df["regret"].astype(float).round(2)
    return df


def load_experiments(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
        experiments = [Experiment.from_dictionary(dct, filepath) for dct in data]

    df = experiments_to_df(experiments)
    return experiments, df


def draw_regrets(logs, show=None, title=None, fig_size=(15, 10)):
    if show is None:
        show = ["master"]
    if title is None:
        title = "Algorithm"

    plt.figure(figsize=fig_size)
    grid = np.arange(logs.total_time - logs.shift)

    if "ideal" in show:
        plt.plot(grid, logs.ideal_losses[logs.shift:].cumsum(),
                     label="Ideal expert cumulative losses", color='green')

    if "master" in show:
        plt.plot(grid, logs.master_losses_all[logs.shift:].cumsum(),
                     label="Master cumulative losses", color='red')

    if "theoretical" in show:
        plt.plot(grid, logs.theoretical_upper[logs.shift:],
                     label="Theoretical upper bound", color='black')

    plt.xlabel("Time")
    plt.ylabel("Regret")
    plt.legend()

    ax = plt.gca()

    bottom, top = ax.get_ybound()
    left, right = ax.get_xbound()
    for gen_idx, gen_stamp in zip(np.r_[logs.indexes, -1], logs.stamps):
        if gen_stamp < logs.shift:
            continue
        plt.axvline(gen_stamp - logs.shift, color='orange', linestyle=':')
        if gen_idx != -1:
            ax.text(x=gen_stamp - logs.shift + 0.005 * (right - left), y=top - 0.12 * (top - bottom),
                              s=f"gen {gen_idx}", color='orange', rotation=15)

    plt.title(title, fontsize=15)
    plt.show()