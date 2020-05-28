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

def plot_vincentized_rt_pdf(exp_data, model_rts, cumulative=False):
    fig, axes = plt.subplots(3, 3, figsize=(10,8), sharex=True, sharey=True)
    conditions = [{'d': d, 'TTA': TTA}
                  for d in sorted(exp_data.d_condition.unique()) 
                  for TTA in sorted(exp_data.tta_condition.unique())]
    q = np.linspace(0.01, 0.99, 10)
#    q = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for (ax, condition) in zip(axes.flatten(), conditions):
        if condition['d'] == 90:
            if condition['TTA'] == 4:
                ax.set_axis_off()
                ax.text(0.5, 0.0, 'TTA=%is' % condition['TTA'], fontsize=18, transform=ax.transAxes, 
                        horizontalalignment='center', verticalalignment='center')
            else:
                ax.text(0.5, 1.01, 'TTA=%is' % condition['TTA'], fontsize=18, transform=ax.transAxes, 
                        horizontalalignment='center', verticalalignment='center')
        if condition['TTA'] == 6:
            ax.text(1.0, 0.5, 'd=%im' % condition['d'], fontsize=18, transform=ax.transAxes, rotation=-90, 
                horizontalalignment='center', verticalalignment='center')
        
        condition_data = exp_data[(exp_data.is_turn_decision) 
                            & (exp_data.d_condition==condition['d']) 
                            & (exp_data.tta_condition==condition['TTA'])]
        if len(condition_data) >= 10:
            condition_quantiles = condition_data.groupby('subj_id').apply(lambda d: np.quantile(a=d.RT, q=q)).mean()
            rt_range = np.linspace(condition_quantiles.min(), condition_quantiles.max(), len(q))
            step = rt_range[1] - rt_range[0]
            rt_grid = np.concatenate([rt_range[:3]-3*step, rt_range, rt_range[-3:]+step*3])
            vincentized_cdf = np.interp(rt_grid, condition_quantiles, q)
            
            if cumulative:
                ax.plot(rt_grid, vincentized_cdf, label='Data', color='C1', ls='', marker='o')
                ax.set_ylim([-0.05, 1.1])
                ax.set_yticks([0.0, 0.5, 1.0])
            else:
                _, data_pdf = np.histogram(condition_data.RT, bins=rt_grid)
                vincentized_pdf = differentiate(t=rt_grid, x=vincentized_cdf)
                ax.plot(rt_grid, vincentized_pdf, label='Data', color='C1')
            
            if not model_rts is None:
                condition_rts = model_rts[(model_rts.subj_id=='all') 
                                        & (model_rts.d_condition==condition['d']) 
                                        & (model_rts.tta_condition==condition['TTA'])]
                ax.plot(condition_rts.t, condition_rts.rt_corr_pdf, label='Model', color='grey')
            
        ax.set_xlabel('')
        ax.set_xlim((0, 1.5))
        sns.despine(offset=5, trim=True)
#        ax.text(0.7, 0.8, str(condition), fontsize=12, transform=ax.transAxes,  
#                horizontalalignment='center', verticalalignment='center')

#    legend_elements = [Patch(facecolor='C1', alpha=0.5, label='Data'),
#                       Line2D([0], [0], color='grey', lw=2, label='Model')]
#     fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(.98, 0.5), fontsize=16, frameon=False)
    fig.text(0.4, 0.04, 'Response time', fontsize=18)
    fig.text(0.04, 0.5, 'CDF' if cumulative else 'PDF', fontsize=18, rotation=90)
    
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

def write_to_csv(directory, filename, array):
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.join(directory, filename), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(array)
       
def fit_model(model, training_data, loss_function):
    training_sample = ddm.Sample.from_pandas_dataframe(df=training_data, 
                                                       rt_column_name='RT', 
                                                       correct_column_name='is_turn_decision')
    return(ddm.fit_adjust_model(sample=training_sample, model=model, 
                                lossfunction=loss_function, suppress_output=True))
