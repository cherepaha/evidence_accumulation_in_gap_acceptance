import models
import pandas as pd
import helper

modelTtaBounds = models.ModelTtaBounds()
exp_data = pd.read_csv('../data/measures.csv', usecols=['subj_id', 'RT', 'is_turn_decision',
                                                    'tta_condition', 'd_condition'])
subjects = exp_data.subj_id.unique()