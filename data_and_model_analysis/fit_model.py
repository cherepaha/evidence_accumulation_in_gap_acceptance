import models
import loss_functions
import pandas as pd
import helper
import os

def fit_model_by_condition(model_no=1, subj_idx=0, n=1, n_training_conditions=9, test_conditions="all"):
    '''
    NB: This script can (and should) be run in parallel in several different python consoles, one subject per console
    model_idx: 1 for the full model described in the paper; 2 for the model with fixed bounds; 3 for "vanilla" DDM
    subj_idx: 0 to 15 to obtain individual fits, or "all" to fit to group-averaged data
    n: number of repeated fits per condition (n>1 can be used to quickly check robustness of model fitting)
    n_training_conditions: defines how many conditions will be included in the training set (4, 8, or 9)
                            For `4`, training data for each condition (TTA, d) will be the decisions where both TTA and d
                            are different from those of the current condition. For `8`, all other conditions will be included.
    test_conditions: "all" to cross-validate model on all nine conditions, or a list of dicts with conditions for which to fit the model
    '''
    print("Model %i, subj idx %s" % (model_no, str(subj_idx)))

    if model_no == 1:
        model = models.ModelDynamicDriftCollapsingBounds()
    if model_no == 2:
        model = models.ModelDynamicDriftFixedBounds()
    if model_no == 3:
        model = models.ModelStaticDriftFixedBounds()
    else:
        ValueError("model_no should be 1, 2, or 3")

    exp_data = pd.read_csv("../data/measures.csv", usecols=["subj_id", "RT", "is_go_decision",
                                                            "tta_condition", "d_condition"])
    subjects = exp_data.subj_id.unique()

    if test_conditions == "all":
        conditions = [{"tta": tta, "d": d}
                      for tta in exp_data.tta_condition.unique()
                      for d in exp_data.d_condition.unique()]
    else:
        conditions = test_conditions

    if subj_idx == "all":
        subj_id = "all"
        subj_data = exp_data
        loss = loss_functions.LossWLSVincent
    else:
        subj_id = subjects[subj_idx]
        subj_data = exp_data[(exp_data.subj_id == subj_id)]
        loss = loss_functions.LossWLS

    output_directory = ("../model_fit_results/model_%i/best_fit_parameters/%s/" %
                        (model_no,
                         "full_data" if n_training_conditions == 9 else "cross_validation_%i" % n_training_conditions))

    file_name = "subj_%s.csv" % (str(subj_id))
    if not os.path.isfile(os.path.join(output_directory, file_name)):
        if n > 1:
            helper.write_to_csv(output_directory, file_name,
                                ["subj_id", "tta_condition", "d_condition", "n", "loss"] + model.param_names,
                                write_mode="w")
        else:
            helper.write_to_csv(output_directory, file_name,
                                ["subj_id", "tta_condition", "d_condition", "loss"] + model.param_names, write_mode="w")

    for i in range(n):
        print("Training conditions: %i" % n_training_conditions)
        print(subj_id)

        if n_training_conditions == 9:
            training_data = subj_data
            print("len(training_data): " + str(len(training_data)))

            fitted_model = helper.fit_model(model.model, training_data, loss)
            if n > 1:
                helper.write_to_csv(output_directory, file_name,
                                    [subj_id, "NA", "NA", i, fitted_model.get_fit_result().value()]
                                    + [float(param) for param in fitted_model.get_model_parameters()])
            else:
                helper.write_to_csv(output_directory, file_name,
                                    [subj_id, "NA", "NA", fitted_model.get_fit_result().value()]
                                    + [float(param) for param in fitted_model.get_model_parameters()])
        else:
            for condition in conditions:
                print(condition)
                if n_training_conditions == 8:
                    training_data = subj_data[
                        (~(subj_data.tta_condition == condition["tta"]) | ~(subj_data.d_condition == condition["d"]))]
                elif n_training_conditions == 4:
                    training_data = subj_data[
                        (~(subj_data.tta_condition == condition["tta"]) & ~(subj_data.d_condition == condition["d"]))]
                else:
                    raise (ValueError("n_training_conditions should be 9, 8, or 4"))
                print("len(training_data): " + str(len(training_data)))
                fitted_model = helper.fit_model(model.model, training_data, loss)
                if n > 1:
                    helper.write_to_csv(output_directory, file_name, [subj_id, condition["tta"], condition["d"], n,
                                                                      fitted_model.get_fit_result().value()]
                                        + [float(param) for param in fitted_model.get_model_parameters()])
                else:
                    helper.write_to_csv(output_directory, file_name, [subj_id, condition["tta"], condition["d"],
                                                                      fitted_model.get_fit_result().value()]
                                        + [float(param) for param in fitted_model.get_model_parameters()])

    return fitted_model


# for subj_idx in [0, 1, 2, 3]:
# for subj_idx in [4, 5, 6, 7]:
# for subj_idx in [8, 9, 10, 11]:
# for subj_idx in [12, 13, 14, 15, "all"]:
#     fit_model_by_condition(model_no=2, n_training_conditions=9, subj_idx=subj_idx)
# for model_no in [2, 3]:

### Cross-validation
#
# test_conditions = [{'tta': 4.0, 'd': 90.0},
#                        {'tta': 4.0, 'd': 150.0},
#                        {'tta': 4.0, 'd': 120.0},
#                        {'tta': 5.0, 'd': 90.0},
#                        {'tta': 5.0, 'd': 150.0},
#                        {'tta': 5.0, 'd': 120.0},
#                        {'tta': 6.0, 'd': 90.0},
#                        {'tta': 6.0, 'd': 150.0},
#                        {'tta': 6.0, 'd': 120.0}]

test_conditions = [{'tta': 5.0, 'd': 120.0},
                   {'tta': 6.0, 'd': 90.0},
                   {'tta': 6.0, 'd': 150.0},
                   {'tta': 6.0, 'd': 120.0}]

fit_model_by_condition(model_no=2, n_training_conditions=8, subj_idx="all", test_conditions=test_conditions)