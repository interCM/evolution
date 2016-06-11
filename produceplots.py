import sys
assert sys.version_info[:2] == (3, 5), "Use python 3.5"
import os, configparser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# These are the "Tableau 20" colors as RGB
TABLEAU20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
    (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
    (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
    (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
    (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
# Scale the RGB values to the [0, 1], which is the format matplotlib accepts
for i in range(len(TABLEAU20)):
    r, g, b = TABLEAU20[i]
    TABLEAU20[i] = (r/255., g/255., b/255.)

COLORS = [c for c in TABLEAU20[::2]]

def get_x(values):
    mi = min(values)
    ma = max(values)
    r = ma - mi
    mi = mi - 0.001*r
    ma = ma + 0.001*r
    return np.linspace(mi, ma, 200)

def get_steady_sate_ind(average_weights):
    deviations = average_weights[1:] - average_weights[:-1]
    min_n = int(0.1*len(average_weights))
    min_dev_ind = np.argmin(np.absolute(np.cumsum(deviations[::-1]))[min_n:])
    if min_dev_ind == 0:
        print("Warning! Probably steady state is detected incorrectly")
    ss_ind = (len(average_weights) - min_n - min_dev_ind)
    return ss_ind


if __name__ == "__main__":
    """ This module is used to create a figure based on the data recorded
    by the Monitor during the evolution.
    """
    assert len(sys.argv) == 2, "Run: $python produceplots.py path/to/run_id"
    run_dir = sys.argv[1]
    assert os.path.isdir(run_dir), "Cannot find run dir"
    run_log_file = os.path.join(run_dir, "run_log.txt")
    assert os.path.isfile(run_log_file), "Run log file not found"
    cfg_file = os.path.join(run_dir, "start.cfg")
    assert os.path.isfile(cfg_file), "Config file not found"

    config = configparser.ConfigParser()
    config.read_file(open(cfg_file))
    pop_sec = [e for e in config.sections() if e.startswith("population")]
    pop_names = [config[p]["name"] for p in pop_sec]

    df = pd.read_table( run_log_file, header=None,
        names=["pop", "par", "val"], sep='\t' )

    assert set(df["pop"]) == set(pop_names)
    pop_steady_state_ind = {}

    fig, axes = plt.subplots(4, figsize=(6,9))

    ss_indixes = []
    for p, color in zip(pop_names, COLORS[:len(pop_names)]):
        y = df.loc[(df["pop"]==p) & (df["par"]=="AV_WEIGHT"), "val"].values
        ss_ind = get_steady_sate_ind(y)
        ss_indixes.append(ss_ind)
        pop_steady_state_ind[p] = ss_ind
        axes[0].plot(range(len(y)), y, color=color, label=p)
    y_lim = axes[0].get_ylim()
    for ss, color in zip(ss_indixes, COLORS[:len(pop_names)]):
        axes[0].plot( [ss, ss], y_lim, color=color, ls='--' )
    axes[0].set_ylim(y_lim)
    axes[0].set_ylabel("Average weight", fontsize=10)
    axes[0].tick_params(axis='x', labelsize=8)
    axes[0].tick_params(axis='y', labelsize=8)
    
    for p, color in zip(pop_names, COLORS[:len(pop_names)]):
        y = df.loc[(df["pop"]==p) & (df["par"]=="AV_GI"), "val"].values
        axes[1].plot(range(len(y)), y, color=color, label=p)
    axes[1].set_ylabel("Average GI (bits)", fontsize=10)
    axes[1].tick_params(axis='x', labelsize=8)
    axes[1].tick_params(axis='y', labelsize=8)

    for p, color in zip(pop_names, COLORS[:len(pop_names)]):
        y = df.loc[(df["pop"]==p) & (df["par"]=="AV_MUT"), "val"].values
        axes[2].plot(range(len(y)), y, color=color, label=p)
    axes[2].set_xlabel("Iteration", fontsize=10)
    axes[2].set_ylabel("Average fraction of\nmutated genome", fontsize=10)
    axes[2].tick_params(axis='x', labelsize=8)
    axes[2].tick_params(axis='y', labelsize=8)

    ax3_min, ax3_max = 100000, -1
    for p, p_sec, color in zip(pop_names, pop_sec, COLORS[:len(pop_names)]):
        rec = df.loc[(df["pop"]==p) & (df["par"]=="WEIGHT_AFTER_REC"),
            "val"].values
        mut = df.loc[(df["pop"]==p) & (df["par"]=="WEIGHT_AFTER_MUT"),
            "val"].values
        sel = df.loc[(df["pop"]==p) & (df["par"]=="WEIGHT_AFTER_SEL"),
            "val"].values
        children_number = config.getint(p_sec, "children_number")
        ss_ind = pop_steady_state_ind[p]
        ss_ind_rec_mut = ss_ind*children_number
        if len(rec) > 0:
            rec = rec[ss_ind_rec_mut:]
            rec_x = get_x(rec)
            rec_y = gaussian_kde(rec)(rec_x)/len(rec)
            axes[3].plot(rec_x, rec_y, color=color, ls='--')
            axes[3].fill(rec_x, rec_y, color=color, alpha=0.1)
        mut = mut[ss_ind_rec_mut:]
        mut_x = get_x(mut)
        mut_y = gaussian_kde(mut)(mut_x)/len(mut)
        axes[3].plot(mut_x, mut_y, color=color, ls='-.')
        axes[3].fill(mut_x, mut_y, color=color, alpha=0.1)
        sel = sel[ss_ind:]
        sel_x = get_x(sel)
        sel_y = gaussian_kde(sel)(sel_x)/len(sel)
        axes[3].plot(sel_x, sel_y, color=color, ls='-')
        axes[3].fill(sel_x, sel_y, color=color, alpha=0.1, label=p)
    axes[3].set_xlabel("Normalized weight", fontsize=10)
    axes[3].set_ylabel("Abundance", fontsize=10)
    axes[3].tick_params(axis='x', labelsize=8)
    axes[3].tick_params(axis='y', labelsize=8)

    for ax in axes:
        ax.legend(loc="best", fontsize=10)
        ax.grid(True)

    plt.tight_layout()
    fig_path = os.path.join(run_dir, "figure.png")
    plt.savefig(fig_path)
    plt.show()
