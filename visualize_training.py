import matplotlib.pyplot as plt

#%%
from config import get_experiment_logs


#%%
def get_x_y_from_logs(log_list):
    x = []
    y = []
    for entry in log_list:
        epoch = entry['epoch']
        value = entry['value']
        x.append(epoch)
        y.append(value)
    return x, y


#%%
# macro_logs = get_experiment_logs('macro_10')
macro_logs = get_experiment_logs('macro_10_150')
# vanilla_logs = get_experiment_logs('vanilla')
vanilla_logs = get_experiment_logs('vanilla_150')


#%%
# the train loss curve
epoch, train_loss = get_x_y_from_logs(macro_logs['epoch_loss'])
plt.plot(epoch, train_loss, label='macro_10')
epoch, train_loss = get_x_y_from_logs(vanilla_logs['epoch_loss'])
plt.plot(epoch, train_loss, label='vanilla')
plt.legend()
plt.show()


#%%
# the train and dev correctness rates
train_cr_ep, train_cr = get_x_y_from_logs(macro_logs['train_correctness_rate'])
dev_cr_ep, dev_cr = get_x_y_from_logs(macro_logs['dev_correctness_rate'])
plt.plot(train_cr_ep, train_cr, label='macro_10 train correct.')
plt.plot(dev_cr_ep, dev_cr, label='macro_10 dev correct.')
train_cr_ep, train_cr = get_x_y_from_logs(vanilla_logs['train_correctness_rate'])
dev_cr_ep, dev_cr = get_x_y_from_logs(vanilla_logs['dev_correctness_rate'])
plt.plot(train_cr_ep, train_cr, label='vanilla train correct.')
plt.plot(dev_cr_ep, dev_cr, label='vanilla dev correct.')
plt.legend()
plt.ylim(0, 1)
plt.show()


#%%
logs = {}
for i in range(1, 10, 2):
    logs[i] = get_experiment_logs(f'macro_{i}')
logs[10] = get_experiment_logs('macro_10')
logs[0] = get_experiment_logs('vanilla')


#%%
def plot_for_different_macro_num(field: str, ylim=None):
    plt.figure()
    for num_macro, curr_logs in logs.items():
        data = curr_logs[field]
        x, y = get_x_y_from_logs(data)
        plt.plot(x, y, label=f'macro={num_macro}')
    plt.title(field)
    if ylim:
        plt.ylim(*ylim)
    plt.legend()
    plt.show()


#%%
# draw training loss plots
plot_for_different_macro_num('epoch_loss')
# draw training correctness plots
plot_for_different_macro_num('train_correctness_rate', (0, 1))
# draw dec correctness plots
plot_for_different_macro_num('dev_correctness_rate', (0, 1))

#%%
logs = {}
no_punc_n_macros = [0, 1, 5, 10]
for n_macro in no_punc_n_macros:
    exp_name = f"no_punc_macro_{n_macro}"
    logs[n_macro] = get_experiment_logs(exp_name)

# draw training loss plots
plot_for_different_macro_num('epoch_loss')
# draw training correctness plots
plot_for_different_macro_num('train_correctness_rate', (0, 1))
# draw dec correctness plots
plot_for_different_macro_num('dev_correctness_rate', (0, 1))


#%%
for i in no_punc_n_macros:
    partitions = ['train', 'dev', 'test']
    s = f"macros={i} final correctness rates: "
    for part in partitions:
        correctness_rate = logs[i][f"{part}_correctness_rate"][-1]['value']
        s += f"{part}={correctness_rate:.2f}, "
    print(s)


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

