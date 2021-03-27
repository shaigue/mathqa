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
