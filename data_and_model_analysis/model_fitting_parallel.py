from model_definitions import LossWLS, ModelTtaBounds
import time
import pandas as pd
import helper

def fit_model_to_subject_data(subj_idx, n=5): 
    modelTtaBounds = ModelTtaBounds()
    exp_data = pd.read_csv('../data/measures.csv', usecols=['subj_id', 'RT', 'is_turn_decision', 
                                                        'tta_condition', 'd_condition'])
    subjects = exp_data.subj_id.unique()
    subj_id = subjects[subj_idx]
    condition = 'all'
    
    directory = '../model_fit_results/'
    file_name = 'model_%s_params_subj_%i.csv' % (modelTtaBounds.model.name[0], subj_id)
    helper.write_to_csv(directory, file_name, ['subj_id', 'i', 'loss'] + modelTtaBounds.param_names)
    
    for i in range(n):
        before = time.time()
        fitted_model = helper.fit_model(subj_id, condition, modelTtaBounds.model, exp_data, LossWLS)
        after= time.time()
    
        print('Subject %i iteration %i: Fitting time: %f minutes' % (subj_id, i, int(after-before)/60))
        
        helper.write_to_csv(directory, file_name, [subj_id, i, fitted_model.get_fit_result().value()] 
                            + fitted_model.get_model_parameters())
