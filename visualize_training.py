import matplotlib.pyplot as plt

#%%
from utils import get_experiment_logs


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
