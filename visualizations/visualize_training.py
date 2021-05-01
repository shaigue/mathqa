import matplotlib.pyplot as plt
from itertools import product
from train_mathqa import TrainLogs
from train_mathqa import get_train_log_path

import config


def get_train_logs_from_exp_name(exp_name: str) -> TrainLogs:
    train_logs_dir = config.get_exp_dir_path(exp_name)
    train_log_path = get_train_log_path(train_logs_dir)
    return TrainLogs.from_json(train_log_path)


def plot_train_loss(train_logs: TrainLogs, exp_name: str):
    epochs, losses = train_logs.get_epoch_value_lists('train', 'loss')
    plt.plot(epochs, losses)
    plt.title(f'{exp_name} train loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


def plot_train_dev_correctness_rate(train_logs: TrainLogs, exp_name: str):
    for part in ['train', 'dev']:
        epochs, correctness_rates = train_logs.get_epoch_value_lists(part, 'correctness_rate')
        plt.plot(epochs, correctness_rates, label=part)
    plt.title(f'{exp_name} train/dev correctness rate')
    plt.legend()
    plt.ylim(0, 1)
    plt.xlabel('epoch')
    plt.ylabel('correctness rate')
    plt.show()


def get_max_correctness_rates(train_logs: TrainLogs) -> dict[str, float]:
    max_cr = {}
    for part in ['train', 'test', 'dev']:
        max_cr[part] = train_logs.get_max_value(part, 'correctness_rate')
    return max_cr


def get_train_dashboard(exp_name: str):
    print(exp_name)
    train_logs = get_train_logs_from_exp_name(exp_name)
    print(f"max_correctness_rates={get_max_correctness_rates(train_logs)}")
    plot_train_loss(train_logs, exp_name)
    plot_train_dev_correctness_rate(train_logs, exp_name)


def overlay_multiple_correctness_rates(exp_names: list[str]):
    for part in ['train', 'dev']:
        for exp_name in exp_names:
            train_logs = get_train_logs_from_exp_name(exp_name)
            epochs, correctness_rates = train_logs.get_epoch_value_lists(part, 'correctness_rate')
            plt.plot(epochs, correctness_rates, label=exp_name)
        plt.title(f'{part} correctness rate during training')
        plt.legend()
        plt.ylim(0, 1)
        plt.xlabel('epoch')
        plt.ylabel('correctness rate')
        plt.show()


def main():
    exp_names = [
        'complexity_higher_than_4',
        'complexity_higher_than_4_macro',
    ]
    overlay_multiple_correctness_rates(exp_names)
    # diff_avg_exps = []
    # for avg_len, macro, suffix in product([2, 3, 4, 5, 6], ['', '_macro'], ['', '_0', '_1']):
    #     diff_avg_exps.append(f'diff_avg_len_{avg_len}{macro}{suffix}')
    # exp_names += diff_avg_exps
    for exp_name in exp_names:
        get_train_dashboard(exp_name)


if __name__ == "__main__":
    main()



# def plot_for_different_macro_num(logs: dict, field: str, ylim=None):
#     plt.figure()
#     for num_macro, curr_logs in logs.items():
#         data = curr_logs[field]
#         gnn_data, y = get_x_y_from_logs(data)
#         plt.plot(gnn_data, y, label=f'macro={num_macro}')
#     plt.title(field)
#     if ylim:
#         plt.ylim(*ylim)
#     plt.legend()
#     plt.show()

#
# logs = {}
# converge_n_macros = [0, 5, 10]
# for n_macro in converge_n_macros:
#     exp_name = f"converge_macro_{n_macro}"
#     logs[n_macro] = config.load_exp_train_log(exp_name)
#
# # draw training loss plots
# plot_for_different_macro_num(logs, 'epoch_loss')
# # draw training correctness plots
# plot_for_different_macro_num(logs, 'train_correctness_rate', (0, 1))
# # draw dec correctness plots
# plot_for_different_macro_num(logs, 'dev_correctness_rate', (0, 1))
#
#
# for i in converge_n_macros:
#     partitions = ['train', 'dev', 'test']
#     s = f"macros={i} final correctness rates: "
#     for part in partitions:
#         correctness_rate = logs[i][f"{part}_correctness_rate"][-1]['value']
#         s += f"{part}={correctness_rate:.4f}, "
#     print(s)
