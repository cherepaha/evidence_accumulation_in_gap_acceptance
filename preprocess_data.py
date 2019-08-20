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

    data.loc[:,'t'] = data.t.groupby(data.index.names).transform(lambda t: (t-t.min()))

    return data

#def get_tta(traj):
#    t0 = traj.iloc[0]

#merge_txt_files()
data = get_data()

traj = data.loc[305,1,2,25]
print(traj.iloc[0])
#tta_info = data.groupby(data.index.names).apply(get_tta)

for idx, traj in data.groupby(data.index.names):
    print(idx)
#    plt.plot(traj.ego_x, traj.ego_y)
#    plt.plot(traj.bot_x, traj.bot_y)
#    plt.figure()
#    plt.plot(traj.t, traj.bot_x)
#    plt.plot(traj.t, traj.bot_y)
#    plt.plot(traj.t, traj.bot_v)
    plt.plot(traj.t, traj.ego_v)
#    plt.plot(traj.t, savgol_filter(traj.ego_ax, window_length=21, polyorder=2))
#    plt.plot(traj.t, traj.throttle)
