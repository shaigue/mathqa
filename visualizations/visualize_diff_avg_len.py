# plot 3 graphs 'train', 'test', 'dev'
# with final correctness rates in y axis
# and avg prog_len in gnn_data
# showing results both with and without macros
from itertools import product
import matplotlib.pyplot as plt

import config

fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(20, 10))
fig.suptitle('average program length and macros',
             fontsize=16)

partitions = ['train', 'test', 'dev']
for i, part in enumerate(partitions):
    ax = axes[i]
    avg_lens = [2, 3, 4, 5, 6]
    suffixes = ['', '_0', '_1']
    x = []
    y_macro = []
    y_no_macro = []
    for avg_len, suffix in product(avg_lens, suffixes):
        x.append(avg_len)
        no_macro_exp = f'diff_avg_len_{avg_len}{suffix}'
        no_macro_logs = config.load_exp_train_log(no_macro_exp)
        y_no_macro.append(config.get_new_correctness_rate(no_macro_logs, part))

        with_macro_exp = f'diff_avg_len_{avg_len}_macro{suffix}'
        with_macro_logs = config.load_exp_train_log(with_macro_exp)
        y_macro.append(config.get_new_correctness_rate(with_macro_logs, part))

    ax.scatter(x, y_macro, label='macro')
    ax.scatter(x, y_no_macro, label='no macro')
    ax.set_title(part, fontsize=20)
    ax.set_xticks(x)
    ax.set_ylim(0.68, 1)
    if i == 0:
        ax.set_ylabel('correctness rate', fontsize=20)
    if i == 1:
        ax.set_xlabel('average #operations', fontsize=20)
    if i == 2:
        ax.legend(fontsize=20)

plt.tight_layout()
plt.show()
