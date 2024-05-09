import numpy as np
import matplotlib.pyplot as plt


class Analysis(object):
    
    def __init__(self, algo):
        self.algo = algo
        self.regret = 0.
        self.total_time = self.algo.total_time
        
        self.workers_num = self.algo.generator.synthesizer.workers_num
        self.pieces_num = self.algo.generator.pieces_num
        self.indexes = self.algo.generator.indexes
        self.stamps = self.algo.generator.stamps

        self.shift = 0
        
        self.segments_losses = np.full((self.pieces_num, self.total_time), np.inf)
        self.best_partition = np.zeros(self.pieces_num)
        self.best_partition_losses = np.zeros(self.total_time)
        self.zero_losses = np.zeros(self.total_time)

        self.theoretical_upper = np.zeros(self.algo.total_time)

    def count_segments_losses(self):
        for seg_num, left, right in zip(
                np.arange(self.pieces_num),
                self.stamps,
                self.stamps[1:]
        ):
            for i in np.arange(1, left):
                self.segments_losses[seg_num, i] = self.algo.experts_losses_all[left:right, i].sum()
            for i in np.arange(left, self.total_time):
                self.segments_losses[seg_num, i] = self.algo.master_losses_all[left:right].sum()
    
        self.best_partition = np.argmin(self.segments_losses[:, 1:], axis=1) + 1
        for seg_num, left, right in zip(
                np.arange(self.pieces_num),
                self.stamps,
                self.stamps[1:]
        ):
            # self.best_partition_losses[left:right] = self.algo.experts_losses_all[left:right, self.best_partition[seg_num]]
            if seg_num < self.workers_num:
                self.best_partition_losses[left:right] = self.algo.master_losses_all[left:right]
            else:
                self.best_partition_losses[left:right] = self.algo.experts_losses_all[left:right, self.best_partition[seg_num]]
    
    def count_theoretical_upper(self):
        k = 0
        for idx, left, right in zip(self.indexes, self.stamps, self.stamps[1:]):
            if left < self.shift:
                continue
            k += 1
            for t in np.arange(left, right):
                self.theoretical_upper[t] = self.theoretical_upper_func(k, t - self.shift)
    
        self.theoretical_upper[self.shift:] += self.best_partition_losses[self.shift:].cumsum()
    
    
    def post_calculations(self, from_start=True):
        if not from_start:
            self.shift = self.stamps[self.workers_num]
    
        self.count_segments_losses()
        self.count_theoretical_upper()
        self.zero_losses = self.algo.loss_func(self.algo.responses, 0)
        self.regret = self.algo.master_losses_all[self.shift:].sum() - self.best_partition_losses[self.shift:].sum()
    
    
    def draw_all(self, from_start=True, show=None, show_experts=None,
                 show_axes=None, height_ratios=None, suptitle=None, fig_size=None):
        shift = 0 if from_start else self.stamps[self.workers_num]
    
        if show is None:
            show = ["master"]
        if show_experts is None:
            show_experts = []
        if show_axes is None:
            show_axes = ["regret"]
        if height_ratios is None:
            height_ratios = [1 for _ in show_axes]
        if suptitle is None:
            suptitle = "Master regret plot"
        if fig_size is None:
            fig_size = (10, 5)
        assert len(show_axes) == len(height_ratios), "Wrong sizes"
    
        text_idx = 0
    
        fig, axes = plt.subplots(len(show_axes), 1, figsize=fig_size, sharex=True, height_ratios=height_ratios)
        ax = [axes] if len(show_axes) == 1 else axes
        grid = np.arange(self.total_time - shift)
    
        try:
            idx = show_axes.index("value")
    
            ax[idx].plot(grid, self.algo.responses[shift:], label="Real responses",
                         color='black')
    
            if "zero" in show:
                ax[idx].plot(grid, np.full(self.total_time - shift, 0),
                             label="Zero expert predictions",
                             color='yellow')
            for expert_num in show_experts:
                ax[idx].plot(grid, self.algo.experts_predictions_all.T[expert_num][shift:],
                             label=f"Expert {expert_num} predictions")
            if "master" in show:
                ax[idx].plot(grid, self.algo.master_predictions_all[shift:],
                             label="Master predictions",
                             color='red')
            ax[idx].set_xlabel("Time")
            ax[idx].set_ylabel("Value")
            ax[idx].xaxis.set_ticks_position('top')
            ax[idx].legend()
    
        except ValueError:
            pass
    
        try:
            idx = show_axes.index("loss")
            text_idx = idx
    
            if "best" in show:
                ax[idx].plot(grid, self.best_partition_losses[shift:], label="Best partition losses",
                             color='green')
            if "zero" in show:
                ax[idx].plot(grid, self.zero_losses[shift:],
                             label="Zero expert losses",
                             color='yellow')
            for expert_num in show_experts:
                ax[idx].plot(grid, self.algo.experts_losses_all.T[expert_num][shift:],
                             label=f"Expert {expert_num} losses")
            if "master" in show:
                ax[idx].plot(grid, self.algo.master_losses_all[shift:], label="Master losses",
                             color='red')
    
            ax[idx].set_ylabel("Loss")
            ax[idx].legend()
    
        except ValueError:
            pass
    
        try:
            idx = show_axes.index("regret")
    
            if "best" in show:
                ax[idx].plot(grid, self.best_partition_losses[shift:].cumsum(),
                             label="Best partition cumulative losses", color='green')
            if "zero" in show:
                ax[idx].plot(grid, self.zero_losses[shift:].cumsum(),
                             label="Zero expert cumulative losses", color='yellow')
    
            for expert_num in show_experts:
                ax[idx].plot(grid, self.algo.experts_losses_all.T[expert_num][shift:].cumsum(),
                             label=f"Expert {expert_num} cumulative losses")
    
            if "master" in show:
                ax[idx].plot(grid, self.algo.master_losses_all[shift:].cumsum(),
                             label="Master cumulative losses", color='red')
    
            if "theoretical" in show:
                ax[idx].plot(grid, self.theoretical_upper[shift:],
                             label="Theoretical upper bound", color='black')
    
            ax[idx].set_xlabel("Time")
            ax[idx].set_ylabel("Cumulative loss")
            ax[idx].legend()
    
        except ValueError:
            pass
    
        bottom, top = ax[text_idx].get_ybound()
        left, right = ax[text_idx].get_xbound()
        for gen_idx, gen_stamp in zip(np.r_[self.indexes, -1], self.stamps):
            if gen_stamp < shift:
                continue
            for i in range(len(show_axes)):
                ax[i].axvline(gen_stamp - shift, color='orange', linestyle=':')
            if gen_idx != -1:
                ax[text_idx].text(x=gen_stamp - shift + 0.005 * (right - left), y=top - 0.12 * (top - bottom),
                                  s=f"gen {gen_idx}", color='orange', rotation=15)
    
        fig.suptitle(suptitle, fontsize=15)
        plt.show()

    def theoretical_upper_func(self, k, t):
        """
        Calculates the theoretical upper bound on the regret of the GMPP algorithm.

        Args:
            k (int): The number of experts that have been activated so far.
            t (int): The current time step.

        Returns:
            float: The theoretical upper bound on the regret.
        """
        if t <= 1:
            return 0
        return (1 / self.algo.eta *
                ((k + 1) * (2 * np.log(np.log(t)) + np.log(self.algo.weight_const)) + (2 * k + 3) * np.log(t)))


def draw_several(from_start=True, logs=None, labels=None, colors=None, title=None, fig_size=(10, 5), loc='upper left'):
    if title is None:
        title = "Master total losses for different algorithms"
    shift = 0 if from_start else logs[0].shift

    plt.figure(figsize=fig_size)

    grid = np.arange(logs[0].total_time - shift)

    for log, label, color in zip(logs, labels, colors):
        plt.plot(grid, log.master_losses_all[shift:].cumsum(), label=label, color=color)

    plt.xlabel("Time")
    plt.ylabel("Cumulative loss")

    plt.legend(loc=loc)

    bottom, top = plt.gca().get_ybound()
    left, right = plt.gca().get_xbound()
    for gen_idx, gen_stamp in zip(np.r_[logs[0].indexes, -1], logs[0].stamps):
        if gen_stamp < shift:
            continue
        plt.axvline(gen_stamp - shift, color='orange', linestyle=':')
        if gen_idx != -1:
            plt.text(x=gen_stamp - shift + 0.005 * (right - left), y=top - 0.12 * (top - bottom),
                              s=f"gen {gen_idx}", color='orange', rotation=15)

    plt.title(title, fontsize=15)
    plt.show()