import matplotlib.pyplot as plt

import config


def get_x_y_from_logs(log_list):
    x = []
    y = []
    for entry in log_list:
        epoch = entry['epoch']
        value = entry['value']
        x.append(epoch)
        y.append(value)
    return x, y


def plot_for_different_macro_num(logs: dict, field: str, ylim=None):
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


logs = {}
converge_n_macros = [0, 5, 10]
for n_macro in converge_n_macros:
    exp_name = f"converge_macro_{n_macro}"
    logs[n_macro] = config.load_exp_train_log(exp_name)

# draw training loss plots
plot_for_different_macro_num(logs, 'epoch_loss')
# draw training correctness plots
plot_for_different_macro_num(logs, 'train_correctness_rate', (0, 1))
# draw dec correctness plots
plot_for_different_macro_num(logs, 'dev_correctness_rate', (0, 1))


for i in converge_n_macros:
    partitions = ['train', 'dev', 'test']
    s = f"macros={i} final correctness rates: "
    for part in partitions:
        correctness_rate = logs[i][f"{part}_correctness_rate"][-1]['value']
        s += f"{part}={correctness_rate:.4f}, "
    print(s)
