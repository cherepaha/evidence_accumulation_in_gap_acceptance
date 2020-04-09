from model_definitions import LossWLS, ModelTtaBounds
import time
import pandas as pd
import helper

def fit_model_by_subject(subj_idx, ndt='uniform', n=5): 
    modelTtaBounds = ModelTtaBounds(ndt=ndt)
    exp_data = pd.read_csv('../data/measures.csv', usecols=['subj_id', 'RT', 'is_turn_decision', 
                                                        'tta_condition', 'd_condition'])
    subjects = exp_data.subj_id.unique()
    subj_id = subjects[subj_idx]
    condition = 'all'
    
    training_data = exp_data[(exp_data.subj_id == subj_id)]
    
    print(subj_id)
    print(condition)
    
    directory = '../model_fit_results/%s/' % (ndt)
    file_name = 'model_%s_params_subj_%i.csv' % (modelTtaBounds.model.name[0], subj_id)
    helper.write_to_csv(directory, file_name, ['subj_id', 'i', 'loss'] + modelTtaBounds.param_names)
    
    for i in range(n):
        before = time.time()
#        fitted_model = helper.fit_model(subj_id, condition, modelTtaBounds.model, exp_data, LossWLS)
        fitted_model = helper.fit_model(modelTtaBounds.model, training_data, LossWLS)        
        after= time.time()
    
        print('Subject %i iteration %i: Fitting time: %f minutes' % (subj_id, i, int(after-before)/60))
        
        helper.write_to_csv(directory, file_name, [subj_id, i, fitted_model.get_fit_result().value()] 
                            + fitted_model.get_model_parameters())


def fit_model_by_condition(subj_idx, cross_validation_operator): 
    '''
    cross_validation_operator defines which conditions will be included in the training set.
    For `and`, training data for each condition (TTA, d) will be the decisions where both TTA and d
    are different from those of the current condition. For `or`, all other conditions will be included.
    '''
    modelTtaBounds = ModelTtaBounds()
    exp_data = pd.read_csv('../data/measures.csv', usecols=['subj_id', 'RT', 'is_turn_decision', 
                                                        'tta_condition', 'd_condition'])
    subjects = exp_data.subj_id.unique()
    conditions = [{'tta': tta, 'd': d} 
                   for tta in exp_data.tta_condition.unique() 
                   for d in exp_data.d_condition.unique()]
        
    subj_id = subjects[subj_idx]
    
    directory = '../model_fit_results/cross_validation/'
    file_name = 'model_%s_cv_%s_subj_%i.csv' % (modelTtaBounds.model.name[0], cross_validation_operator, subj_id)
    helper.write_to_csv(directory, file_name, ['subj_id', 'tta', 'd', 'loss'] + modelTtaBounds.param_names)
    
    for condition in conditions:
        print(cross_validation_operator)
        print(subj_id)
        print(condition)        
            
        f = lambda x,y: (~x & ~y) if cross_validation_operator=='and' else (~x | ~y)
        include_in_training_set = f(exp_data.tta_condition==condition['tta'], exp_data.d_condition==condition['d'])
       
        training_data = exp_data[(exp_data.subj_id == subj_id) & include_in_training_set]
        print(len(training_data))
        
        fitted_model = helper.fit_model(modelTtaBounds.model, training_data, LossWLS)
    
        helper.write_to_csv(directory, file_name, [subj_id, condition['tta'], condition['d'], 
                                                   fitted_model.get_fit_result().value()] 
                                                    + fitted_model.get_model_parameters())
