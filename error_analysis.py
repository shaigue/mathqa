"""This script is for analyzing errors of the model"""

import config


#%%
from mathqa_processing import MathQAManager

exp_no_macro_name = 'converge_macro_0'
exp_yes_macro_name = 'converge_macro_10'


#%%
logs_no_macro = config.load_exp_train_log(exp_no_macro_name)
logs_yes_macro = config.load_exp_train_log(exp_yes_macro_name)


#%%
# lets look at the dev errors
no_macro_dev_error_reports = logs_no_macro['dev_per_sample_report']
yes_macro_dev_error_reports = logs_yes_macro['dev_per_sample_report']


#%%
# just look at each of their errors
from train_mathqa import get_manager

no_macro_manager = get_manager()
yes_macro_manager = get_manager(macro_file=config.get_macro_file(10))

no_macro_errors = []
yes_macro_errors = []

for i in range(len(no_macro_dev_error_reports)):
    no_macro_report = no_macro_dev_error_reports[i]
    if no_macro_report['error_type'] != 'no_error':
        no_macro_errors.append(i)
    yes_macro_report = yes_macro_dev_error_reports[i]
    if yes_macro_report['error_type'] != 'no_error':
        yes_macro_errors.append(i)

print(f"no_macro_errors: {len(no_macro_errors)}")
print(f"yes_macro_errors: {len(yes_macro_errors)}")


#%%
# pretty print the produced programs and the original programs
from pprint import pprint


def pp_errors(manager: MathQAManager, error_report: list[dict], errors: list[int]):
    to_print = []
    bad_macro = 0
    for i in errors:
        generated = manager.code_vectorizer.token_list_to_string(error_report[i]['generated_tokens'])
        datapoint = manager.get_datapoint('dev', i)
        original = str(datapoint.program)
        if 'macro' in generated and 'macro' not in original:
            bad_macro += 1
        if 'macro' in original:
            print('***macro***')
        to_print.append({'generated': generated, 'original': original})
    print(f"bad_macro={bad_macro}")
    pprint(to_print)


#%%
only_yes_macro_errors = set(yes_macro_errors).difference(no_macro_errors)
only_no_macro_errors = set(no_macro_errors).difference(yes_macro_errors)
print(f"only macro errors: {len(only_yes_macro_errors)}")
print(f"only no macro errors: {len(only_no_macro_errors)}")


#%%
# print only_no_macro_errors
pp_errors(yes_macro_manager, no_macro_dev_error_reports, only_no_macro_errors)

#%%
# print only_no_macro_errors
pp_errors(yes_macro_manager, yes_macro_dev_error_reports, only_yes_macro_errors)
