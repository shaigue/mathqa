import matplotlib.pyplot as plt
import json
import config


#%%
def get_experiment_logs(experiment_name: str):
    filename = 'train_log.json'
    experiment_dir = config.TRAINING_LOGS_DIR / experiment_name / filename
    with experiment_dir.open('r') as f:
        experiment_logs = json.load(f)
    return experiment_logs


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
macro_logs = get_experiment_logs('macro_10')
vanilla_logs = get_experiment_logs('vanilla')


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
