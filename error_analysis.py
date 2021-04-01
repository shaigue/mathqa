"""This is a script for exploring the errors that different models make."""


#%%
# lets look at some samples in "train" that 10 macros got it and 0 macros didn't
entry1 = 10
entry2 = 0
part = 'train'
field = f'{part}_per_sample_report'
reports1 = logs[entry1][field]
reports2 = logs[entry2][field]


#%%
# how much of each was value errors and how much was syntax error
def syntax_value_error_analysis(reports: list[dict]) -> tuple[float, float]:
    n_syntax, n_value = 0, 0
    for rep in reports:
        error_type = rep['error_type']
        if error_type == 'syntax':
            n_syntax += 1
        if error_type == 'value':
            n_value += 1
    total_error = n_syntax + n_value
    p_syntax = n_syntax / total_error
    p_value = n_value / total_error
    return p_syntax, p_value


#%%
p_syntax1, p_value1 = syntax_value_error_analysis(reports1)
p_syntax2, p_value2 = syntax_value_error_analysis(reports2)
print(f"{entry1} syntax={p_syntax1}, value={p_value1}")
print(f"{entry2} syntax={p_syntax2}, value={p_value2}")


#%%
# find places where entry1 succeeded and entry2 failed
succ1_fail2 = {}
for i in range(len(reports1)):
    rep1 = reports1[i]
    rep2 = reports2[i]
    print(rep1, rep2)
    error1 = rep1['error_type'] != 'no_error'
    error2 = rep2['error_type'] != 'no_error'
    if not error1 and error2:
        succ1_fail2[i] = {'rep1': rep1, 'rep2': rep2}

print(succ1_fail2)
#%%
fail_reports = [rep['rep2'] for rep in succ1_fail2.values()]
p_syntax_diff, p_value_diff = syntax_value_error_analysis(fail_reports)
print(f"diff syntax={p_syntax_diff}, value={p_value_diff}")


#%%

from train_mathqa import get_manager

manager = get_manager()
# find the average of all the dataset length
# find the average of all the error of 2 length
# find the average of correct 1 and error of 2 length
succ1_fail2_indices = set(succ1_fail2.keys())
succ1_fail2_lens = []
fail2_indices = set()
for i, rep in enumerate(reports2):
    if rep['error_type'] != 'no_error':
        fail2_indices.add(i)
fail2_lens = []
all_lens = []
for i, datapoint in enumerate(manager.iter_dataset(part)):
    code_len = len(datapoint.code_token_indices)
    all_lens.append(code_len)
    if i in succ1_fail2_indices:
        succ1_fail2_lens.append(code_len)
    if i in fail2_indices:
        fail2_lens.append(code_len)

from statistics import mean
fail2_avg_len = mean(fail2_lens)
succ1_fail2_avg_len = mean(succ1_fail2_lens)
avg_len = mean(all_lens)
print(f"all={avg_len}, succ1_fail2={succ1_fail2_avg_len}, fail2={fail2_avg_len}")

