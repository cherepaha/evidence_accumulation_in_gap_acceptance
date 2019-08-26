import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

# merge all text files in data_path together
def merge_txt_files(data_path='data'):
    dfs = []
    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)
        if file_path.endswith('.txt'):
            print(file_path)
            dfs.append(pd.read_csv(file_path, sep='\t'))
    df_concat = pd.concat(dfs)
    df_concat.to_csv(os.path.join('raw_data_merged.txt'), index=False, sep='\t')

def get_RT(traj):
    if sum(traj.ego_v<1):
        idx_bot_spawn = (traj.ego_v<0.9).to_numpy().nonzero()[0][0]+5 # assume 50 ms delay for bot spawn
        throttle = traj.iloc[idx_bot_spawn:, traj.columns.get_loc('throttle')]
        idx_response = idx_bot_spawn + (throttle>0.01).to_numpy().nonzero()[0][0]
        RT = traj.t[idx_response] - traj.t[idx_bot_spawn]
    else:
        idx_bot_spawn = -1
        idx_response = -1
        RT = -1
    return pd.Series({'idx_bot_spawn': idx_bot_spawn,
                      'idx_response': idx_response,
                      'RT': RT})

def get_data(data_file='raw_data_merged.txt'):
    data = pd.read_csv(data_file, sep='\t', index_col=['subj_id', 'session', 'route', 'intersection'])

    data.t /= 1000

    # we are only intersted in left turns
    data = data[data.turn_direction==1]

    # only consider parts of the trajectories near the target intersection
    data = data[abs(data.distance_to_intersection)<10]

    apply_filter = lambda traj: savgol_filter(traj, window_length=21, polyorder=2, axis=0)
    cols_to_smooth = ['ego_vx', 'ego_vy', 'ego_ax', 'bot_ay', 'bot_vx', 'bot_vy', 'bot_ax', 'bot_ay']
    data.loc[:, cols_to_smooth] = (data.loc[:, cols_to_smooth].groupby(data.index.names).transform(apply_filter))

    data['ego_v'] = np.sqrt(data.ego_vx**2 + data.ego_vy**2)
    data['bot_v'] = np.sqrt(data.bot_vx**2 + data.bot_vy**2)

    data['ego_a'] = np.sqrt(data.ego_ax**2 + data.ego_ay**2)
    data['bot_a'] = np.sqrt(data.bot_ax**2 + data.bot_ay**2)

    measures = data.groupby(data.index.names).apply(get_RT)

    conditions = data.loc[:,['tta_condition', 'd_condition', 'v_condition']].groupby(data.index.names).first()
    measures = measures.join(conditions)

    data.loc[:,'t'] = data.t.groupby(data.index.names).transform(lambda t: (t-t.min()))

    return data, measures

#merge_txt_files()
data, measures = get_data()
subjects = data.index.get_level_values(0).unique()

measures['turn_before'] = measures.RT < measures.tta_condition
measures = measures[measures.RT>0]
measures = measures.reset_index()

#for idx, group in measures.groupby('subj_id'):
sns.pointplot(data=measures, x='d_condition', y='turn_before', hue='subj_id')

sns.pointplot(data=measures, x='d_condition', y='RT', hue='subj_id')

traj = data.loc[subjects[0],1,1,19]

#data = data.join(measures)
#for idx, traj in data.groupby(data.index.names):
#    print(idx)
##    plt.plot(traj.ego_x, traj.ego_y)
##    plt.plot(traj.bot_x, traj.bot_y)
#    plt.figure()
##    plt.plot(traj.t, traj.bot_x)
##    plt.plot(traj.t, traj.bot_y)
#
#    plt.plot(traj.t, traj.ego_v, color='grey')
#    idx_bot_spawn = traj.iloc[0, traj.columns.get_loc('idx_bot_spawn')]
#    idx_response = traj.iloc[0, traj.columns.get_loc('idx_response')]
##
#    plt.plot(traj.t[idx_bot_spawn], traj.throttle[idx_bot_spawn], marker='o', ls='')
#    plt.plot(traj.t[idx_response], traj.throttle[idx_response], marker='x', ls='')


#    plt.plot(traj.t[idx_bot_spawn:], traj.bot_x[idx_bot_spawn:])
#    plt.plot(traj.t, savgol_filter(traj.ego_ax, window_length=21, polyorder=2))
#    plt.plot(traj.t, traj.throttle)
