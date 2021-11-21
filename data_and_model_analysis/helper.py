import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ddm
import os
import csv
from matplotlib.lines import Line2D

def merge_csv(directory):
    fout = open(directory+'_parameters_fitted.csv','w+')
    header_written = False
    for i, file_name in enumerate(os.listdir(directory)):
        file_path = os.path.join(directory, file_name)
        if file_path.endswith('.csv'):
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

def plot_var_by_subject(data, fit_results_path, var, ylabel):
    model_measures = pd.read_csv(os.path.join(fit_results_path, 'measures.csv'))
    d_conditions = np.sort(data.d_condition.unique())
    subjects = data.subj_id.unique()
    
    fig, axes = plt.subplots(2, 4, figsize=(12,6), sharex=True, sharey=True)
    ax = axes[0][0]
    ax.set_title('all subjects', fontsize=16)
    sns.pointplot(ax=axes[0][0], data=data, 
                      x='tta_condition', y=var, hue='d_condition', join=False, dodge=0.05,
                      markers=['o', 's', '^'], hue_order=d_conditions, scale=0.8, errwidth=2)

    for d_condition, marker in zip(d_conditions, ['o', 's', '^']):
        ax.plot([0, 1, 2], model_measures.loc[model_measures.d_condition==d_condition, var], zorder=0.1)

    ax.legend().remove()
    ax.set_xlabel('')
    ax.set_ylabel('')

    for subj_id, ax in zip(subjects, axes.flatten()[1:]):
        ax.set_title(subj_id, fontsize=16)
        if not ((subj_id == 616) & (var=='RT')):
            sns.pointplot(ax=ax, data=data[data.subj_id==subj_id], 
                      x='tta_condition', y=var, hue='d_condition', join=False, dodge=0.05,
                      markers=['o', 's', '^'], scale=1, errwidth=2)
        else:
            sns.pointplot(ax=ax, data=data[data.subj_id==subj_id], 
                      x='tta_condition', y=var, hue='d_condition', join=False, dodge=0.05,
                      markers=['s', '^'], palette=['C1', 'C2'], scale=1, errwidth=2)

        subj_fit_results_path = os.path.join(fit_results_path, str(subj_id))
        subj_model_measures = pd.read_csv(os.path.join(subj_fit_results_path, 'measures.csv'))
        for d_condition, marker in zip(d_conditions, ['o', 's', '^']):
            ax.plot([0, 1, 2], subj_model_measures.loc[subj_model_measures.d_condition==d_condition, var], 
                    zorder=0.1)

        ax.legend().remove()
        ax.set_xlabel('')
        ax.set_ylabel('')
    plt.tight_layout()
    sns.despine(offset=5, trim=True)

    legend_elements = [Line2D([0], [0], color='C0', marker='o', lw=0, label='Data, d=90m'),
                       Line2D([0], [0], color='C1', marker='s', lw=0, label='Data, d=120m'),
                       Line2D([0], [0], color='C2', marker='^', lw=0, label='Data, d=150m'),
                       Line2D([0], [0], color='grey', label='Model fits')]

    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(1.15, 0.55), fontsize=16, frameon=False)

    fig.text(0.4, -0.05, 'time-to-arrival (TTA), s', fontsize=18)
    fig.text(-0.03, 0.3, ylabel, fontsize=18, rotation=90)

    return fig, axes

def plot_var_by_subject_v2(data, model_measures_path, var, ylabel):
    model_measures = pd.read_csv(model_measures_path)
    d_conditions = np.sort(data.d_condition.unique())
    subjects = data.subj_id.unique()
    
    fig, axes = plt.subplots(2, 4, figsize=(12,6), sharex=True, sharey=True)
#    ax = axes[0][0]
#    ax.set_title('all subjects', fontsize=16)
#    sns.pointplot(ax=axes[0][0], data=data, 
#                      x='tta_condition', y=var, hue='d_condition', join=False, dodge=0.05,
#                      markers=['o', 's', '^'], hue_order=d_conditions, scale=0.8, errwidth=2)
#
#    for d_condition, marker in zip(d_conditions, ['o', 's', '^']):
#        ax.plot([0, 1, 2], model_measures.loc[model_measures.d_condition==d_condition, var], zorder=0.1)
#
#    ax.legend().remove()
#    ax.set_xlabel('')
#    ax.set_ylabel('')

    for subj_id, ax in zip(subjects, axes.flatten()[:-1]):
        ax.set_title(subj_id, fontsize=16)
        if not ((subj_id == 616) & (var=='RT')):
            sns.pointplot(ax=ax, data=data[data.subj_id==subj_id], 
                      x='tta_condition', y=var, hue='d_condition', join=False, dodge=0.05,
                      markers=['o', 's', '^'], scale=1, errwidth=2)
        else:
            sns.pointplot(ax=ax, data=data[data.subj_id==subj_id], 
                      x='tta_condition', y=var, hue='d_condition', join=False, dodge=0.05,
                      markers=['s', '^'], palette=['C1', 'C2'], scale=1, errwidth=2)

#        subj_fit_results_path = os.path.join(fit_results_path, str(subj_id))
#        subj_model_measures = pd.read_csv(os.path.join(subj_fit_results_path, 'measures.csv'))
        for d_condition, marker in zip(d_conditions, ['o', 's', '^']):
            ax.plot([0, 1, 2], model_measures.loc[(model_measures.subj_id==subj_id) & 
                                                  (model_measures.d_condition==d_condition), var], 
                    zorder=0.1)

        ax.legend().remove()
        ax.set_xlabel('')
        ax.set_ylabel('')
    plt.tight_layout()
    sns.despine(offset=5, trim=True)

    legend_elements = [Line2D([0], [0], color='C0', marker='o', lw=0, label='Data, d=90m'),
                       Line2D([0], [0], color='C1', marker='s', lw=0, label='Data, d=120m'),
                       Line2D([0], [0], color='C2', marker='^', lw=0, label='Data, d=150m'),
                       Line2D([0], [0], color='grey', label='Model fits')]

    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(1.15, 0.55), fontsize=16, frameon=False)

    fig.text(0.4, -0.05, 'time-to-arrival (TTA), s', fontsize=18)
    fig.text(-0.03, 0.3, ylabel, fontsize=18, rotation=90)

    return fig, axes

def differentiate(t, x):
    # To be able to reasonably calculate derivatives at the end-points of the trajectories,
    # I append three extra points before and after the actual trajectory, so we get N+6
    # points instead of N       
    x = np.append(x[0]*np.ones(3), np.append(x, x[-1]*np.ones(3)))
    
    # Time vector is also artificially extended by equally spaced points
    # Use median timestep to add dummy points to the time vector
    timestep = np.median(np.diff(t))
    t = np.append(t[0]-np.arange(1,4)*timestep, np.append(t, t[-1]+np.arange(1,4)*timestep))

    # smooth noise-robust differentiators, see: 
    # http://www.holoborodko.com/pavel/numerical-methods/ \
    # numerical-derivative/smooth-low-noise-differentiators/#noiserobust_2
    v = (1*(x[6:]-x[:-6])/((t[6:]-t[:-6])/6) + 
         4*(x[5:-1] - x[1:-5])/((t[5:-1]-t[1:-5])/4) + 
         5*(x[4:-2] - x[2:-4])/((t[4:-2]-t[2:-4])/2))/32
    
    return v

def write_to_csv(directory, filename, array, write_mode='a'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.join(directory, filename), write_mode, newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(array)
       
def fit_model(model, training_data, loss_function):
    training_sample = ddm.Sample.from_pandas_dataframe(df=training_data, 
                                                       rt_column_name='RT', 
                                                       correct_column_name='is_turn_decision')
    return ddm.fit_adjust_model(sample=training_sample, model=model, lossfunction=loss_function, verbose=False)
