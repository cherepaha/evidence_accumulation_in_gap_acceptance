import os
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

def merge_txt_files(data_path):
    dfs = []
    raw_data_path = os.path.join(data_path, 'raw')
    for file in os.listdir(raw_data_path):
        file_path = os.path.join(raw_data_path, file)
        if file_path.endswith('.txt'):
            print(file_path)
            dfs.append(pd.read_csv(file_path, sep='\t'))
    df_concat = pd.concat(dfs)
    df_concat.to_csv(os.path.join(data_path, 'raw_data_merged.txt'), index=False, sep='\t')

def get_measures(traj):
    '''
    This function extracts dependent variables and some other useful things from an individual trajectory.    
    The time of bot spawn is the first time bot_a is less than 10% of its max value (after the ego car slows
    down to 1 m/s). 
    '''   
    if sum(traj.ego_v<1.0):
        idx_slowdown = (traj.ego_v<1.0).to_numpy().nonzero()[0][0]
        idx_bot_spawn = traj.bot_x.to_numpy().nonzero()[0][0] + (traj[idx_slowdown:].bot_a<1).to_numpy().nonzero()[0][0]
        throttle = traj.iloc[idx_bot_spawn:, traj.columns.get_loc('throttle')]
        idx_response = idx_bot_spawn + (throttle>0).to_numpy().nonzero()[0][0]
        RT = traj.t.values[idx_response] - traj.t.values[idx_bot_spawn]
        idx_min_distance = idx_bot_spawn + np.argmin(traj.d_ego_bot[idx_bot_spawn:].values)
        min_distance = min(traj.d_ego_bot[idx_bot_spawn:].values)
    else:
        # if the driver never decelerated to <1m/s, the bot did not spawn
        idx_bot_spawn = -1
        idx_response = -1
        idx_min_distance = -1
        min_distance = -1
        RT = -1
    return pd.Series({'idx_bot_spawn': idx_bot_spawn,
                      'idx_response': idx_response,
                      'idx_min_distance': idx_min_distance,
                      'min_distance': min_distance,
                      'RT': RT})

def get_data(data_file='raw_data_merged.txt'):
    data = pd.read_csv(data_file, sep='\t', index_col=['subj_id', 'session', 'route', 'intersection_no'])
    
    data.loc[:,'t'] = data.t.groupby(data.index.names).transform(lambda t: (t-t.min()))
    
    # we are only intersted in left turns
    data = data[data.turn_direction==1]

    # only consider the data recorded within 10 meters of each intersection
    data = data[abs(data.ego_distance_to_intersection)<10]

    # smooth the time series by filtering out the noise using Savitzky-Golay filter
    apply_filter = lambda traj: savgol_filter(traj, window_length=21, polyorder=2, axis=0)
    cols_to_smooth = ['ego_x', 'ego_y', 'ego_vx', 'ego_vy', 'ego_ax', 'ego_ay', 
                      'bot_x', 'bot_y', 'bot_vx', 'bot_vy', 'bot_ax', 'bot_ay']
    data.loc[:, cols_to_smooth] = (data.loc[:, cols_to_smooth].groupby(data.index.names).transform(apply_filter))

    # calculate absolute values of speed and acceleration
    data['ego_v'] = np.sqrt(data.ego_vx**2 + data.ego_vy**2)
    data['bot_v'] = np.sqrt(data.bot_vx**2 + data.bot_vy**2)    
    data['ego_a'] = np.sqrt(data.ego_ax**2 + data.ego_ay**2)
    data['bot_a'] = np.sqrt(data.bot_ax**2 + data.bot_ay**2)
    
    # calculate actual distance between the ego vehicle and the bot, and current tta for each t
    data['d_ego_bot'] = np.sqrt((data.ego_x - data.bot_x)**2 + (data.ego_y - data.bot_y)**2)    
    data['tta'] = data.d_ego_bot/data.bot_v
    
    # get the DVs and helper variables
    measures = data.groupby(data.index.names).apply(get_measures)
#     print(measures.groupby(['subj_id', 'session', 'route']).count())
    
    data = data.join(measures)    
    
    # RT is -1 if a driver didn't stop and the bot did not appear at the intersection; we discard these trials
    print('Number of discarded trials: %i' % (len(measures[measures.RT<=0])))
    
    data = data[data.RT>0]
    measures = measures[measures.RT>0]    
    
    # add the condition information to the measures dataframe for further analysis
    conditions = data.loc[:,['tta_condition', 'd_condition', 'v_condition']].groupby(data.index.names).first()
    measures = measures.join(conditions)
    
    return data, measures
    

data_path='../data'

#merge_txt_files(data_path)
data, measures = get_data(os.path.join(data_path, 'raw_data_merged.txt'))

# is_turn_decision is calculated based on the minimum distance ever observed between ego and bot during the interaction.
# If the cars were no closer to each other than the lane width (3.5m + 1.5 margin), we count this as a turn decision.
# Based on visual inspection of animations for all trials, this works in all but one trials.
measures['is_turn_decision'] = measures.min_distance > 5

# In some trials this criterion wouldn't work, because e.g. a subject might hit a post after turning left, 
# so that the bot is less than 5m away from the ego car after the turn. We need to check this and fix manually
# measures.loc[(305, 1, 1, 8), ['is_turn_decision']] = True

# add column 'decision' for nicer visualization
measures['decision'] = 'Wait'
measures.loc[measures.is_turn_decision, ['decision']] = 'Turn'

measures.to_csv(os.path.join(data_path, 'measures.csv'), index=True)
data.to_csv(os.path.join(data_path, 'processed_data.csv'), index=True)