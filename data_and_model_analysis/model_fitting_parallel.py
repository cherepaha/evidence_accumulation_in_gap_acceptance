import model_definitions
import pandas as pd
import helper
#
#def fit_model_by_subject(subj_idx, ndt='gaussian', n=5): 
#    modelTtaBounds = model_definitions.ModelTtaBounds(ndt=ndt)
#    exp_data = pd.read_csv('../data/measures.csv', usecols=['subj_id', 'RT', 'is_turn_decision', 
#                                                        'tta_condition', 'd_condition'])
#    subjects = exp_data.subj_id.unique()
#    subj_id = subjects[subj_idx]
#    condition = 'all'
#    
#    training_data = exp_data[(exp_data.subj_id == subj_id)]
#    
#    print(subj_id)
#    print(condition)
#    
#    directory = '../model_fit_results/%s_ndt/' % (ndt)
#    file_name = 'model_%s_params_subj_%i.csv' % (modelTtaBounds.model.name[0], subj_id)
#    helper.write_to_csv(directory, file_name, ['subj_id', 'i', 'loss'] + modelTtaBounds.param_names)
#    
#    for i in range(n):
#        before = time.time()
##        fitted_model = helper.fit_model(subj_id, condition, modelTtaBounds.model, exp_data, LossWLS)
#        fitted_model = helper.fit_model(modelTtaBounds.model, training_data, LossWLS)        
#        after= time.time()
#    
#        print('Subject %i iteration %i: Fitting time: %f minutes' % (subj_id, i, int(after-before)/60))
#        
#        helper.write_to_csv(directory, file_name, [subj_id, i, fitted_model.get_fit_result().value()] 
#                            + fitted_model.get_model_parameters())

def fit_model_by_condition(subj_idx=0, n=1, training_conditions='all'): 
    '''
    training_conditions defines how many conditions will be included in the training set.
    For `4`, training data for each condition (TTA, d) will be the decisions where both TTA and d
    are different from those of the current condition. For `8`, all other conditions will be included.
    '''
    modelTtaBounds = model_definitions.ModelTtaBounds(ndt='gaussian')
    exp_data = pd.read_csv('../data/measures.csv', usecols=['subj_id', 'RT', 'is_turn_decision', 
                                                        'tta_condition', 'd_condition'])
    subjects = exp_data.subj_id.unique()
    conditions = [{'tta': tta, 'd': d} 
                    for tta in exp_data.tta_condition.unique() 
                    for d in exp_data.d_condition.unique()]
    
    if subj_idx == 'all':
        subj_id = 'all'
        subj_data = exp_data 
        loss = model_definitions.LossWLSVincent
    else:
        subj_id = subjects[subj_idx]
        subj_data = exp_data[(exp_data.subj_id == subj_id)]
        loss = model_definitions.LossWLS
        
    directory = '../model_fit_results/cross_validation_%s/' % (training_conditions)
        
    file_name = 'subj_%s.csv' % (str(subj_id))
    if n>1:
        helper.write_to_csv(directory, file_name, ['subj_id', 'tta', 'd', 'n', 'loss'] + modelTtaBounds.param_names)
    else:
        helper.write_to_csv(directory, file_name, ['subj_id', 'tta', 'd', 'loss'] + modelTtaBounds.param_names)
    
    for i in range(n):
        print('Training conditions: %s' % (training_conditions))
        print(subj_id)
        
        if training_conditions=='all':
            training_data = subj_data
            print('len(training_data): ' + str(len(training_data)))
            
            fitted_model = helper.fit_model(modelTtaBounds.model, training_data, loss)
            if n>1:
                helper.write_to_csv(directory, file_name, [subj_id, 'NA', 'NA', n, fitted_model.get_fit_result().value()] 
                                                        + fitted_model.get_model_parameters())
            else:
                helper.write_to_csv(directory, file_name, [subj_id, 'NA', 'NA', fitted_model.get_fit_result().value()] 
                                                        + fitted_model.get_model_parameters())
        else:
            for condition in conditions:
                print(condition)
                if training_conditions=='8':
                    training_data = subj_data[(~(subj_data.tta_condition==condition['tta']) | ~(subj_data.d_condition==condition['d']))]
                elif training_conditions=='4':
                    training_data = subj_data[(~(subj_data.tta_condition==condition['tta']) & ~(subj_data.d_condition==condition['d']))]
                else:
                    raise(ValueError('training_conditions should be one of ["all", "8", "4"]'))    
                print('len(training_data): ' + str(len(training_data)))                
                fitted_model = helper.fit_model(modelTtaBounds.model, training_data, loss)            
                if n>1:
                    helper.write_to_csv(directory, file_name, [subj_id, condition['tta'], condition['d'], n, 
                                                           fitted_model.get_fit_result().value()] 
                                                            + fitted_model.get_model_parameters())
                else:
                    helper.write_to_csv(directory, file_name, [subj_id, condition['tta'], condition['d'], 
                                                           fitted_model.get_fit_result().value()] 
                                                            + fitted_model.get_model_parameters())
