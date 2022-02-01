import numpy as np
import pandas as pd
import scipy
import ddm
import os
import csv


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


def get_mean_sem(data, var="RT", n_cutoff=2):
    mean = data.groupby("tta_condition")[var].mean()
    sem = data.groupby("tta_condition")[var].apply(lambda x: scipy.stats.sem(x, axis=None, ddof=0))
    n = data.groupby("tta_condition").size()
    data_mean_sem = pd.DataFrame({"mean": mean, "sem": sem, "n": n}, index=mean.index)
    data_mean_sem = data_mean_sem[data_mean_sem.n > n_cutoff]

    return data_mean_sem
