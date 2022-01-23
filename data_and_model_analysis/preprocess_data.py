import os
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter


def merge_txt_files(data_path):
    dfs = []
    raw_data_path = os.path.join(data_path, "raw")
    for file in os.listdir(raw_data_path):
        file_path = os.path.join(raw_data_path, file)
        if file_path.endswith(".txt"):
            print(file_path)
            dfs.append(pd.read_csv(file_path, sep="\t"))
    df_concat = pd.concat(dfs)
    df_concat.to_csv(os.path.join(data_path, "raw_data_merged.txt"), index=False, sep="\t")


def get_measures(traj):
    """
    This function extracts dependent variables and some other useful measures from an individual trajectory.
    """
    print(traj.name)

    # -1 is assigned as the default value so that if the algorithms below do not trigger, we exclude the trial later on
    idx_bot_spawn = -1
    idx_response = -1
    idx_min_distance = -1
    min_distance = -1
    RT = -1

    # if the driver decelerated to <1m/s and the bot did spawn, we can calculate the RT
    if (traj.ego_v < 1.0).any() and (abs(traj.bot_x) > 0.0).any():
        # detect when the ego vehicle first slowed down to <1m/s
        idx_slowdown = (traj.ego_v < 1.0).to_numpy().nonzero()[0][0]
        # to find out when the bot appeared (t1), detect when (after the vehicle started slowing down) the bot's
        # acceleration (which should be 0) first decreases below 1m/s^2
        idx_bot_spawn = idx_slowdown + (traj[idx_slowdown:].bot_a < 1).to_numpy().nonzero()[0][0]

        # finding the moment when the driver responded (t2),
        throttle = traj.throttle
        # by default, the index of the first non-zero value of throttle after the bot spawned is marked as idx_response
        # this accurately determines idx_response if the throttle was not pressed at the time of the bot spawn,
        idx_response = idx_bot_spawn + (throttle.iloc[idx_bot_spawn:] > 0).to_numpy().nonzero()[0][0]

        # however, if throttle was pressed when the bot was spawned, we need to calculate idx_response differently
        # based on the visual inspection of such trials, we define idx_response as the onset time of the next
        # throttle press after the one already in process at the time the bot spawned
        # to calculate it, we first check if there were any zero throttle values after the bot spawned
        if (throttle.iloc[idx_bot_spawn:] == 0).any():
            # if so, we find the next zero value of throttle (idx_first_zero)
            idx_first_zero = idx_bot_spawn + (throttle.iloc[idx_bot_spawn:] == 0).to_numpy().nonzero()[0][0]
            # and then idx_response is defined as the index of the first non-zero value after that
            idx_response = idx_first_zero + (throttle.iloc[idx_first_zero:] > 0).to_numpy().nonzero()[0][0]

        # RT = t2 - t1
        RT = traj.t.values[idx_response] - traj.t.values[idx_bot_spawn]

        # the minimum distance between the cars is used to determine which decision the driver has made
        idx_min_distance = idx_bot_spawn + np.argmin(traj.d_ego_bot[idx_bot_spawn:].values)
        min_distance = min(traj.d_ego_bot[idx_bot_spawn:].values)

    return pd.Series({"idx_bot_spawn": idx_bot_spawn,
                      "idx_response": idx_response,
                      "idx_min_distance": idx_min_distance,
                      "min_distance": min_distance,
                      "RT": RT})


def get_processed_data(data_file="raw_data_merged.txt"):
    data = pd.read_csv(data_file, sep="\t", index_col=["subj_id", "session", "route", "intersection_no"])

    # transforming timestamp so that every trajectory starts at t=0
    data.loc[:, "t"] = data.t.groupby(data.index.names).transform(lambda t: (t - t.min()))

    # we are only interested in left turns
    data = data[data.turn_direction == 1]

    # only consider the data recorded within 10 meters of each intersection
    data = data[abs(data.ego_distance_to_intersection) < 10]

    # smooth the time series by filtering out noise using the Savitzky-Golay filter
    apply_filter = lambda traj: savgol_filter(traj, window_length=21, polyorder=2, axis=0)
    cols_to_smooth = ["ego_x", "ego_y", "ego_vx", "ego_vy", "ego_ax", "ego_ay",
                      "bot_x", "bot_y", "bot_vx", "bot_vy", "bot_ax", "bot_ay"]
    data.loc[:, cols_to_smooth] = (data.loc[:, cols_to_smooth].groupby(data.index.names).transform(apply_filter))

    # calculate absolute values of speed and acceleration
    data["ego_v"] = np.sqrt(data.ego_vx ** 2 + data.ego_vy ** 2)
    data["bot_v"] = np.sqrt(data.bot_vx ** 2 + data.bot_vy ** 2)
    data["ego_a"] = np.sqrt(data.ego_ax ** 2 + data.ego_ay ** 2)
    data["bot_a"] = np.sqrt(data.bot_ax ** 2 + data.bot_ay ** 2)

    # calculate actual distance between the ego vehicle and the bot, and current tta for each t
    data["d_ego_bot"] = np.sqrt((data.ego_x - data.bot_x) ** 2 + (data.ego_y - data.bot_y) ** 2)
    data["tta"] = data.d_ego_bot / data.bot_v

    # get the DVs and helper variables
    measures = data.groupby(data.index.names).apply(get_measures)
    print(measures.groupby(["subj_id", "session", "route"]).size())

    # merging the measures into the dynamics dataframe to manipulate the latter more conveniently
    data = data.join(measures)

    # RT is defined as -1 if a driver didn't stop and the bot did not appear at the intersection; we discard these trials
    print("Number of discarded trials: %i" % (len(measures[measures.RT <= 0])))
    print(measures[measures.RT <= 0].groupby(["subj_id"]).size())

    # is_go_decision is calculated based on the minimum distance observed between ego and bot during the interaction.
    # If the cars were no closer to each other than the lane width (3.5m + 1.5 margin), we count this as a go
    # decision. Based on visual inspection of animations for all trials, this works in the vast majority of trials
    # except for collisions
    measures["is_go_decision"] = measures.min_distance > 5

    # In some trials this criterion doesn't work because the ego vehicle collided with the bot. In these trials the
    # bot is less than 5m away from the ego vehicle even though the participant made the go decision. These trials
    # were identified based on visual inspection of the trajectories
    collision_traj_ids = [(129, 1, 1, 5), (280, 1, 2, 16), (421, 1, 1, 5), (421, 1, 1, 20), (421, 1, 2, 3),
                          (421, 1, 2, 21), (421, 2, 8, 8), (525, 1, 2, 7), (755, 1, 1, 8), (853, 1, 1, 13),
                          (853, 1, 3, 9), (853, 1, 4, 10), (853, 2, 6, 13)]

    # We mark these trials with a flag just in case
    measures.loc[:, ["is_collision"]] = False
    measures.loc[collision_traj_ids, ["is_collision"]] = True
    # ... and change the go decision flag from False to True (because they are otherwise marked as "stay" trials)
    measures.loc[collision_traj_ids, ["is_go_decision"]] = True

    # add column "decision" for convenience of visualization
    measures["decision"] = "Stay"
    measures.loc[measures.is_go_decision, ["decision"]] = "Go"

    # add the condition information to the measures dataframe for further analysis
    conditions = data.loc[:, ["tta_condition", "d_condition", "v_condition"]].groupby(data.index.names).first()
    measures = measures.join(conditions)

    measures[measures.RT <= 0].to_csv(os.path.join(data_path, "measures_excluded.csv"), index=True)
    data[data.RT <= 0].to_csv(os.path.join(data_path, "processed_data_excluded.csv"), index=True)

    data = data[data.RT > 0]
    measures = measures[measures.RT > 0]

    return data, measures


data_path = "../data"

# merge_txt_files(data_path)
data, measures = get_processed_data(os.path.join(data_path, "raw_data_merged.txt"))

measures.to_csv(os.path.join(data_path, "measures.csv"), index=True)
data.to_csv(os.path.join(data_path, "processed_data.csv"), index=True)
