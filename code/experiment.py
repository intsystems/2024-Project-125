import numpy as np
import pandas as pd
import json
import dataclasses_json
import dataclasses


from synthesizer import Synthesizer
from generator import Generator
from algorithm import Algorithm

from hypers import weight_hypers
from hypers import alpha_hypers

# Fixed params

series_type = "default"
from_start = False
a, b = -40., 40.
seeds = 5 * np.arange(2, 4) + 11
# train_window = 10
noise_var = 1


dim = 10
low, high = -10, 10
clip = (a, b)
workers_num = 3

length = 2000
lower_bound, upper_bound = 100, 400
alternating=True


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
    experts_predictions_all: np.array
    masters_predictions_all: np.array

    experts_losses_all: np.array
    master_losses_all: np.array
    zero_losses: np.array
    weights_all: np.array
    ideal_losses: np.array
    theoretical_upper: np.array


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class Experiment:
    random_seed: int

    key_w: str
    key_a: str
    train_window: int
    noise_var: float

    params: Params
    logs: Logs
    regret: float

    @classmethod
    def from_dictionary(cls, exp_dict):
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
            experts_predictions_all=np.array(exp_dict['logs']['experts_predictions_all']),
            masters_predictions_all=np.array(exp_dict['logs']['masters_predictions_all']),
            experts_losses_all=np.array(exp_dict['logs']['experts_losses_all']),
            master_losses_all=np.array(exp_dict['logs']['master_losses_all']),
            zero_losses=np.array(exp_dict['logs']['zero_losses']),
            weights_all=np.array(exp_dict['logs']['weights_all']),
            ideal_losses=np.array(exp_dict['logs']['ideal_losses']),
            theoretical_upper=np.array(exp_dict['logs']['theoretical_upper']),
        )

        return Experiment(
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


def run_experiments(windows, interesting_w, interesting_a, filepath=None):
    interesting = set()

    for key_w in interesting_w:
        interesting.add((key_w, "default"))

    for key_a in interesting_a:
        interesting.add(("simple_101", key_a))

    experiments = []

    for rs in seeds:
        for train_window in windows:
            synt = Synthesizer(series_type, dim=dim, low=low, high=high, clip=clip,
                               noise_var=noise_var, workers_num=workers_num, random_seed=rs)
            gen = Generator(series_type, synt)
            gen.generate(length=length, from_start=from_start,
                         lower_bound=lower_bound, upper_bound=upper_bound, alternating=True)

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
                                workers_num=workers_num,
                                length=length,
                                lower_bound=lower_bound,
                                upper_bound=upper_bound,
                                alternating=alternating)

                logs = Logs(total_time=algo.total_time,
                            shift=algo.shift,
                            indexes=gen.indexes,
                            stamps=gen.stamps,
                            experts_predictions_all=algo.experts_predictions_all,
                            masters_predictions_all=algo.master_predictions_all,
                            experts_losses_all=algo.experts_losses_all,
                            master_losses_all=algo.master_losses_all,
                            zero_losses=algo.zero_losses,
                            weights_all=algo.weights_all,
                            ideal_losses=algo.ideal_losses,
                            theoretical_upper=algo.theoretical_upper,
                            )

                experiment = Experiment(random_seed=rs,
                                        key_w=key_w,
                                        key_a=key_a,
                                        train_window=train_window,
                                        noise_var=noise_var,
                                        params=params,
                                        logs=logs,
                                        regret=algo.regret)

                experiments.append(experiment)

                del algo
            del gen
            del synt

    df = experiments_to_df(experiments)

    if filepath is not None:
        df.to_csv(filepath, sep='\t')
        # with open(filepath, 'w') as f:
            # json.dump(experiments, f, cls=EnhancedJSONEncoder)

    return experiments, df


def experiments_to_df(experiments):
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

    renaming = {}
    for i in list(df):
        if i < 3:
            renaming[i] = labels[i]
        else:
            renaming[i] = f"random_{i - 2}"

    df = df.set_index(np.arange(df.index.size)).rename(columns=renaming)
    ordered_repr = (sorted(df['train_window'].unique()) +
                    [wh.repr for wh in weight_hypers.values()] +
                    [ah.repr for ah in alpha_hypers.values()])
    df = df.sort_values(labels, key=lambda col: col.apply(lambda x: ordered_repr.index(x)))
    df.insert(3, "mean", df.iloc[:, 3:].mean(axis=1))
    df["mean"] = df["mean"].astype(float).round(2)
    return df


def load_experiments(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
        experiments = [Experiment.from_dictionary(dct) for dct in data]

    df = experiments_to_df(experiments)

    return experiments, df

