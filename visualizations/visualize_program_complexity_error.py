from matplotlib import pyplot as plt
import numpy as np

import config


def complexity_correctness_plot():
    """Plots some plots"""
    # get the training logs
    exp_name = 'converge_macro_0'
    train_log = config.load_exp_train_log(exp_name)

    # get the original data
    data = math_qa.load_all_dataset(config.MATHQA_DIR)

    # collect the (complexity, error) pairs
    complexity_error_pairs = []
    for part in ['train', 'test', 'dev']:
        entries = data[part]
        reports = train_log[f'{part}_per_sample_report']
        for i, entry in enumerate(entries):
            complexity = log1_num_tokens(entry.linear_formula)
            error = False if reports[i]['error_type'] == 'no_error' else True
            complexity_error_pairs.append((complexity, error))

    # find the partition of the complexity into bins
    max_complexity = -np.inf
    min_complexity = np.inf
    for complexity, error in complexity_error_pairs:
        if complexity > max_complexity:
            max_complexity = complexity
        if complexity < min_complexity:
            min_complexity = complexity

    # partition into bins
    n_bins = 14
    bins = np.linspace(min_complexity, max_complexity, n_bins, endpoint=True)
    total_count = np.zeros_like(bins)
    error_count = np.zeros_like(bins)
    for complexity, error in complexity_error_pairs:
        bin_i = 0
        while bin_i < n_bins - 1 and complexity > bins[bin_i + 1]:
            bin_i += 1
        total_count[bin_i] += 1
        if error:
            error_count[bin_i] += 1
    bin_error_p = error_count / total_count
    plt.plot(bins, bin_error_p)
    plt.title('error probability / program complexity')
    plt.xlabel('program complexity = log2(1 + #tokens)')
    plt.ylabel('error probability')
    plt.ylim(0, 1.1)
    plt.show()
    plt.hist([ce[0] for ce in complexity_error_pairs], bins=bins, log=True)
    plt.title('program complexity histogram')
    plt.xlabel('program complexity = log2(1 + #tokens)')
    plt.ylabel('count - log scaled')
    plt.show()


if __name__ == "__main__":
    complexity_correctness_plot()
