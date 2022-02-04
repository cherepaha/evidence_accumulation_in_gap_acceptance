import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import ddm
import os
import csv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def merge_csv(directory):
    fout = open(directory + "_parameters_fitted.csv", "w+")
    header_written = False
    for i, file_name in enumerate(os.listdir(directory)):
        file_path = os.path.join(directory, file_name)
        if file_path.endswith(".csv"):
            f = open(file_path)
            if header_written:
                # skip the header for the first row
                next(f)
            for line in f:
                fout.write(line)
            f.close()
            header_written = True
            print(file_path)
    fout.close()


def differentiate(t, x):
    # To be able to reasonably calculate derivatives at the end-points of the trajectories,
    # I append three extra points before and after the actual trajectory, so we get N+6
    # points instead of N       
    x = np.append(x[0] * np.ones(3), np.append(x, x[-1] * np.ones(3)))

    # Time vector is also artificially extended by equally spaced points
    # Use median timestep to add dummy points to the time vector
    timestep = np.median(np.diff(t))
    t = np.append(t[0] - np.arange(1, 4) * timestep, np.append(t, t[-1] + np.arange(1, 4) * timestep))

    # smooth noise-robust differentiators, see: 
    # http://www.holoborodko.com/pavel/numerical-methods/ \
    # numerical-derivative/smooth-low-noise-differentiators/#noiserobust_2
    v = (1 * (x[6:] - x[:-6]) / ((t[6:] - t[:-6]) / 6) +
         4 * (x[5:-1] - x[1:-5]) / ((t[5:-1] - t[1:-5]) / 4) +
         5 * (x[4:-2] - x[2:-4]) / ((t[4:-2] - t[2:-4]) / 2)) / 32

    return v


def write_to_csv(directory, filename, array, write_mode="a"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.join(directory, filename), write_mode, newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(array)


def fit_model(model, training_data, loss_function):
    training_sample = ddm.Sample.from_pandas_dataframe(df=training_data,
                                                       rt_column_name="RT",
                                                       correct_column_name="is_go_decision")
    return ddm.fit_adjust_model(sample=training_sample, model=model, lossfunction=loss_function, verbose=False)


def get_psf_ci(data):
    # psf: psychometric function
    # ci: dataframe with confidence intervals for probability per coherence
    tta_conditions = np.sort(data.tta_condition.unique())

    psf = np.array([len(data[data.is_go_decision & (data.tta_condition == tta_condition)])
                    / len(data[data.tta_condition == tta_condition])
                    if len(data[(data.tta_condition == tta_condition)]) > 0 else np.NaN
                    for tta_condition in np.sort(data.tta_condition.unique())])

    ci = pd.DataFrame(psf, columns=["p_go"], index=tta_conditions)

    n = [len(data[(data.tta_condition == tta_condition)]) for tta_condition in tta_conditions]
    ci["ci_l"] = ci["p_go"] - np.sqrt(psf * (1 - psf) / n)
    ci["ci_r"] = ci["p_go"] + np.sqrt(psf * (1 - psf) / n)

    return ci.reset_index().rename(columns={"index": "tta_condition"})


def get_mean_sem(data, var="RT", groupby_var="tta_condition", n_cutoff=2):
    mean = data.groupby(groupby_var)[var].mean()
    sem = data.groupby(groupby_var)[var].apply(lambda x: scipy.stats.sem(x, axis=None, ddof=0))
    n = data.groupby(groupby_var).size()
    data_mean_sem = pd.DataFrame({"mean": mean, "sem": sem, "n": n}, index=mean.index)
    data_mean_sem = data_mean_sem[data_mean_sem.n > n_cutoff]

    return data_mean_sem


def plot_all_subj_p_go(ax, exp_measures, d_condition, marker, color, marker_offset=0):
    between_subj_mean = exp_measures[(exp_measures.d_condition == d_condition)].groupby(
        ["subj_id", "tta_condition"]).mean()
    data_subj_d_measures = get_mean_sem(between_subj_mean.reset_index(), var="is_go_decision", n_cutoff=2)
    ax.errorbar(data_subj_d_measures.index + marker_offset, data_subj_d_measures["mean"],
                yerr=data_subj_d_measures["sem"],
                ls="", marker=marker, ms=9, color=color)


def plot_subj_p_go(ax, exp_measures, d_condition, subj_id, marker, color):
    data_subj_d_measures = exp_measures[(exp_measures.subj_id == subj_id) & (exp_measures.d_condition == d_condition)]
    psf_ci = get_psf_ci(data_subj_d_measures)
    ax.plot(psf_ci.tta_condition, psf_ci.p_go, ls="", marker=marker, ms=9, color=color, zorder=10)
    ax.vlines(x=psf_ci.tta_condition, ymin=psf_ci.ci_l, ymax=psf_ci.ci_r, color=color, zorder=10)


def plot_subj_rt(ax, exp_measures, d_condition, subj_id, marker, color, marker_offset=0):
    if subj_id == "all":
        between_subj_mean = exp_measures[
            (exp_measures.d_condition == d_condition) & (exp_measures.is_go_decision)].groupby(
            ["subj_id", "tta_condition"]).mean()
        measures = between_subj_mean.reset_index()
    else:
        measures = exp_measures[(exp_measures.subj_id == subj_id) & (exp_measures.d_condition == d_condition) & (
            exp_measures.is_go_decision)]

    if len(measures) > 0:
        measures_mean_sem = get_mean_sem(measures, var="RT", n_cutoff=2)
        ax.errorbar(measures_mean_sem.index + marker_offset, measures_mean_sem["mean"], yerr=measures_mean_sem["sem"],
                    ls="", marker=marker, ms=9, color=color)


def plot_condition_vincentized_dist(ax, condition, condition_data, kind="cdf"):
    # colors = dict(zip([90,120,150], [plt.cm.viridis(r) for r in np.linspace(0.1,0.7,3)]))
    # markers={90: "o", 120: "s", 150: "^"}
    #     q = [0.1, 0.3, 0.5, 0.7, 0.9]
    q = np.linspace(0.01, 0.99, 15)
    condition_quantiles = condition_data.groupby("subj_id").apply(lambda d: np.quantile(a=d.RT, q=q)).mean()

    rt_range = np.linspace(condition_quantiles.min(), condition_quantiles.max(), len(q))
    step = rt_range[1] - rt_range[0]
    rt_grid = np.concatenate([rt_range[:3] - 3 * step, rt_range, rt_range[-3:] + step * 3])
    vincentized_cdf = np.interp(rt_grid, condition_quantiles, q, left=0, right=1)
    # vincentized_pdf = differentiate(rt_grid, vincentized_cdf)

    ax.plot(rt_grid, vincentized_cdf, label="Data", color="grey", ls="", ms=9, marker="o")
    ax.set_ylim([-0.05, 1.1])
    ax.set_yticks([0.0, 0.5, 1.0])


def decorate_axis(ax, condition):
    if (((condition["d"] == 90) & (condition["TTA"] == 6))
            | ((condition["d"] == 90) & (condition["TTA"] == 5))
            | ((condition["d"] == 120) & (condition["TTA"] == 4))):
        ax.text(0.5, 1.02, "TTA=%is" % condition["TTA"], fontsize=16, transform=ax.transAxes,
                horizontalalignment="center", verticalalignment="center")

    if condition["TTA"] == 6:
        ax.text(1.0, 0.5, "d=%im" % condition["d"], fontsize=16, transform=ax.transAxes, rotation=-90,
                horizontalalignment="center", verticalalignment="center")


def plot_vincentized_dist(fig, axes, exp_data, model_rts, model_color="black", plot_data=True):
    conditions = [{"d": d, "TTA": TTA}
                  for d in sorted(exp_data.d_condition.unique())
                  for TTA in sorted(exp_data.tta_condition.unique())]

    for (ax, condition) in zip(axes.flatten(), conditions):
        condition_data = exp_data[(exp_data.is_go_decision)
                                  & (exp_data.d_condition == condition["d"])
                                  & (exp_data.tta_condition == condition["TTA"])]
        if len(condition_data) >= 25:
            # Group-averaged data
            if plot_data:
                plot_condition_vincentized_dist(ax, condition, condition_data)

            # Model
            if not model_rts is None:
                condition_rts = model_rts[(model_rts.subj_id == "all")
                                          & (model_rts.d_condition == condition["d"])
                                          & (model_rts.tta_condition == condition["TTA"])]
                ax.plot(condition_rts.t, condition_rts.rt_corr_distr, color=model_color, alpha=0.8,
                        lw=2)  # , color="C%i" % (model_no-1))
        else:
            ax.set_axis_off()

        if plot_data:
            decorate_axis(ax, condition)

            ax.set_xlabel("")
            ax.set_xlim((0, 1.5))
            sns.despine(offset=5, trim=True)

    if plot_data:
        fig.text(0.43, 0.04, "Response time, s", fontsize=16)
        fig.text(0.04, 0.15, "Cumulative distribution function", fontsize=16, rotation=90)

    return fig, axes

def plot_cross_validation(exp_data, model_measures):
    model_measures = model_measures[(model_measures.tta_condition>=4.0) & (model_measures.tta_condition<=6.0)]

    d_conditions = [90, 120, 150]
    colors = [plt.cm.viridis(r) for r in np.linspace(0.1,0.7,len(d_conditions))]
    markers=["o", "s", "^"]
    marker_size=9
    marker_offset = 0.05

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6,3))

    for d_condition, marker, color in zip(d_conditions, markers, colors):
        model_subj_d_measures = model_measures[(model_measures.subj_id=="all")  & (model_measures.d_condition==d_condition)]
        # Model
        ax1.plot(model_subj_d_measures.tta_condition+marker_offset, model_subj_d_measures["is_go_decision"],
                    color=color, label=d_condition, ls="--", lw=1, marker=marker, ms=marker_size, fillstyle="none")
        ax2.plot(model_subj_d_measures.tta_condition+marker_offset, model_subj_d_measures["RT"],
                color=color, label=d_condition, ls="--", lw=1, marker=marker, ms=marker_size, fillstyle="none")

        # Data
        plot_all_subj_p_go(ax1, exp_data, d_condition, marker, color, -marker_offset)
        plot_subj_rt(ax2, exp_data, d_condition, "all", marker, color, -marker_offset)

    fig.text(0.35, -0.05, "Time-to-arrival (TTA), s", fontsize=16)

    ax1.set_xticks([4, 5, 6])
    ax2.set_xticks([4, 5, 6])

    ax1.legend().remove()
    ax2.legend().remove()

    ax1.set_ylabel("Probability of go", fontsize=16)
    ax2.set_ylabel("Response time", fontsize=16)

    ax1.set_ylim((0.0, 1.0))
    ax2.set_ylim((0.3, 0.8))

    sns.despine(offset=5, trim=True)
    plt.tight_layout()

    legend_elements = ([Line2D([0], [0], color=color, marker=marker, ms=marker_size, lw=1, ls="--", fillstyle="none", label="Model predictions,")
                           for d_condition, marker, color in zip(d_conditions, markers, colors)]
                       + [Line2D([0], [0], color=color, marker=marker, ms=marker_size, lw=0, label="data, d=%im" % (d_condition))
                           for d_condition, marker, color in zip(d_conditions, markers, colors)])

    fig.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=16, handlelength=1.5, columnspacing=0.2,
               frameon=False, ncol=2)

    return fig
