import matplotlib.pyplot as plt
from math_qa.math_qa import load_dataset
import config
import numpy as np
import time


#%%
from macro_experiments import get_n_samples, select_samples, get_n_ops


# plot a histogram for train / test/ dev for the number of op_list in the programs
def plot_plots():
    for part in ['train', 'test', 'dev']:
        entries = load_dataset(part, config.MATHQA_DIR)
        program_lens = [get_n_ops(entry) for entry in entries]
        bins = np.arange(1, 55, 1)
        plt.hist(x=program_lens, bins=bins)
        plt.title(f"{part} num. operations histogram")
        plt.show()
        print(f"{part} count={len(program_lens)}, mean={np.mean(program_lens)}, std={np.std(program_lens)}")


#%%



#%%
avg_lens = [2, 3, 4, 5, 6]
frac = 1 / 3

for avg_len in avg_lens:
    for part in ['train', 'test', 'dev']:
        start = time.time()
        print(f"part={part}, avg_len={avg_len}")

        dataset = load_dataset(part, config.MATHQA_DIR)
        n_total_samples = len(dataset)
        n_samples = get_n_samples(n_total_samples, frac)
        n_ops = [get_n_ops(entry) for entry in dataset]
        subset_indices = select_samples(n_ops, n_samples, avg_len)

        print(f"n_samples={n_samples} out of={n_total_samples}, got samples={len(subset_indices)}")
        subset_lens = [n_ops[i] for i in subset_indices]
        lens_mean = np.mean(subset_lens)
        lens_std = np.std(subset_lens)
        print(f"mean={lens_mean}, std={lens_std}")
        end = time.time()
        print(f"time={end-start:.2f}")
