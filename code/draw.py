import numpy as np
from matplotlib import pyplot as plt


def draw_all(logs, show=None, show_experts=None,
             show_axes=None, height_ratios=None, suptitle=None, fig_size=(15, 10)):
    """
    Visualizes the performance of the AA algorithm,
        including predictions, losses, regrets, and theoretical bounds.

    Args:
        show (list, optional): A list of elements to display on the plots (e.g., "master", "zero", "ideal").
            Defaults to ["master"].
        show_experts (list, optional): A list of expert indices to display on the plots. Defaults to [].
        show_axes (list, optional): A list of axes to display on the plots (e.g., "value", "loss", "regret").
            Defaults to ["regret"].
        height_ratios (list, optional): A list of height ratios for the subplots.
            Defaults to [1 for _ in show_axes].
        suptitle (string, optional): Title for the plot
        fig_size (tuple, optional): Figure size. Defaults to (15,10).

    Returns:
        tuple: A tuple containing the figure and axes objects.
    """

    if show is None:
        show = ["master"]
    if show_experts is None:
        show_experts = []
    if show_axes is None:
        show_axes = ["regret"]
    if height_ratios is None:
        height_ratios = [1 for _ in show_axes]
    if suptitle is None:
        suptitle = "Algorithm"
    assert len(show_axes) == len(height_ratios), "Wrong sizes"

    text_idx = 0

    fig, axes = plt.subplots(len(show_axes), 1, figsize=fig_size, sharex=True, height_ratios=height_ratios)
    ax = [axes] if len(show_axes) == 1 else axes
    grid = np.arange(logs.total_time - logs.shift)

    try:
        idx = show_axes.index("value")

        ax[idx].plot(grid, logs.responses[logs.shift:], label="Real responses",
                     color='black')

        if "zero" in show:
            ax[idx].plot(grid, np.full(logs.total_time - logs.shift, 0),
                         label="Zero expert predictions",
                         color='yellow')
        for expert_num in show_experts:
            ax[idx].plot(grid, logs.experts_predictions_all.T[expert_num][logs.shift:],
                         label=f"Expert {expert_num} predictions")
        if "master" in show:
            ax[idx].plot(grid, logs.master_predictions_all[logs.shift:],
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

        if "ideal" in show:
            ax[idx].plot(grid, logs.ideal_losses[logs.shift:], label="Ideal expert losses",
                         color='green')
        if "zero" in show:
            ax[idx].plot(grid, logs.zero_losses[logs.shift:],
                         label="Zero expert losses",
                         color='yellow')
        for expert_num in show_experts:
            ax[idx].plot(grid, logs.experts_losses_all.T[expert_num][logs.shift:],
                         label=f"Expert {expert_num} losses")
        if "master" in show:
            ax[idx].plot(grid, logs.master_losses_all[logs.shift:], label="Master losses",
                         color='red')

        ax[idx].set_ylabel("Loss")
        ax[idx].legend()

    except ValueError:
        pass

    try:
        idx = show_axes.index("regret")

        if "ideal" in show:
            ax[idx].plot(grid, logs.ideal_losses[logs.shift:].cumsum(),
                         label="Ideal expert cumulative losses", color='green')
        if "zero" in show:
            ax[idx].plot(grid, logs.zero_losses[logs.shift:].cumsum(),
                         label="Zero expert cumulative losses", color='yellow')

        for expert_num in show_experts:
            ax[idx].plot(grid, logs.experts_losses_all.T[expert_num][logs.shift:].cumsum(),
                         label=f"Expert {expert_num} cumulative losses")

        if "master" in show:
            ax[idx].plot(grid, logs.master_losses_all[logs.shift:].cumsum(),
                         label="Master cumulative losses", color='red')

        if "theoretical" in show:
            ax[idx].plot(grid, logs.theoretical_upper[logs.shift:],
                         label="Theoretical upper bound", color='black')

        ax[idx].set_xlabel("Time")
        ax[idx].set_ylabel("Regret")
        ax[idx].legend()

    except ValueError:
        pass

    bottom, top = ax[text_idx].get_ybound()
    left, right = ax[text_idx].get_xbound()
    for gen_idx, gen_stamp in zip(np.r_[logs.indexes, -1], logs.stamps):
        if gen_stamp < logs.shift:
            continue
        for i in range(len(show_axes)):
            ax[i].axvline(gen_stamp - logs.shift, color='orange', linestyle=':')
        if gen_idx != -1:
            ax[text_idx].text(x=gen_stamp - logs.shift + 0.005 * (right - left), y=top - 0.12 * (top - bottom),
                              s=f"gen {gen_idx}", color='orange', rotation=15)

    fig.suptitle(suptitle, fontsize=15)
    plt.show()