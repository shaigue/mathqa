converge_experiments = {i: f'converge_macro_{i}' for i in [0, 5, 10]}
many_macros_experiments = {i: f'many_macros_{i}' for i in [21, 41, 61, 81, 100]}
# experiment_names = {}
# experiment_names.update(converge_experiments)
# experiment_names.update(many_macros_experiments)


#%%
import config
converge_logs = {}
for i, exp_name in converge_experiments.items():
    train_log = config.load_exp_train_log(exp_name)
    converge_logs[i] = train_log

many_macros_logs = {}
for i, exp_name in many_macros_experiments.items():
    train_log = config.load_exp_train_log(exp_name)
    many_macros_logs[i] = train_log


#%%
# just get the final train, dev, test correctness rates
def get_old_correctness_rate(train_log: dict, part: str) -> float:
    return train_log[f'{part}_correctness_rate'][-1]['value']


def get_new_correctness_rate(train_log: dict, part: str) -> float:
    d = train_log[part]['correctness_rate']
    k, v = d.popitem()
    d[k] = v
    return v


#%%
correctness_rates = {}
for part in ['train', 'test', 'dev']:
    correctness_rates[part] = {}
    for n_macro, train_log in converge_logs.items():
        correctness_rates[part][n_macro] = get_old_correctness_rate(train_log, part)
    for n_macro, train_log in many_macros_logs.items():
        correctness_rates[part][n_macro] = get_new_correctness_rate(train_log, part)


#%%
import matplotlib.pyplot as plt

for part in ['train', 'test', 'dev']:
    n_macros = []
    cr = []
    for k, v in correctness_rates[part].items():
        n_macros.append(k)
        cr.append(v)
    plt.plot(n_macros, cr, label=part)
    # plt.title(part)
    # plt.ylim(0.6, 1)
    # plt.show()
plt.legend()
plt.title('correctness rate / number of macros substituted')
plt.show()

