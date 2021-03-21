import json

logs_file = 'training_logs.json'
with open(logs_file, 'r') as f:
    logs = json.load(f)

print(list(logs.keys()))


#%%
import matplotlib.pyplot as plt


#%%
plt.figure()


def get_x_y_from_logs(log_list):
    x = []
    y = []
    for entry in log_list:
        epoch = entry['epoch']
        value = entry['value']
        x.append(epoch)
        y.append(value)
    return x, y

x, y = get_x_y_from_logs(logs['epoch_loss'])
plt.plot(x, y)
plt.show()


#%%
plt.figure()
x, y = get_x_y_from_logs(logs['train_correctness_rate'])
plt.plot(x, y, label="train correct.")
x, y = get_x_y_from_logs(logs['dev_correctness_rate'])
plt.plot(x, y, label="dev correct.")
plt.legend()
plt.show()
