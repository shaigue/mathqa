# plot 3 graphs 'train', 'test', 'dev'
# with final correctness rates in y axis
# and avg prog_len in x
# showing results both with and without macros
import matplotlib.pyplot as plt

import config

fig, axes = plt.subplots(1, 3)

partitions = ['train', 'test', 'dev']

for i, part in enumerate(partitions):
    ax = axes[i]
    # TODO: add 5, 6
    x = [2, 3, 4, ]
    y_macro = []
    y_no_macro = []
    for avg_len in x:
        no_macro_exp = f'diff_avg_len_{avg_len}'
        no_macro_logs = config.load_exp_train_log(no_macro_exp)
        y_no_macro.append(config.get_new_correctness_rate(no_macro_logs, part))

        with_macro_exp = f'diff_avg_len_{avg_len}_macro'
        with_macro_logs = config.load_exp_train_log(with_macro_exp)
        y_macro.append(config.get_new_correctness_rate(with_macro_logs, part))

    ax.plot(x, y_macro, label='macro')
    ax.plot(x, y_no_macro, label='no macro')
    ax.set_title(part)
    ax.set_ylim(0.7, 1)
    ax.legend()

plt.show()
