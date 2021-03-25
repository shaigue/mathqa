from utils import get_experiment_logs
from math_qa.dataset import load_dataset, RawMathQAEntry
import config
from statistics import mean

# TODO: categorize the errors to syntax errors and value errors
# TODO: find the 4 sets - both correct, both right, macro correct and vanilla incorrect, and vice versa
# TODO: iterate through a debugger through one of those errors
# TODO: analyze the results with the more epochs


def per_sample_correctness(experiment_name: str, partition: str):
    macro_10_logs = get_experiment_logs(experiment_name)
    return macro_10_logs[f'{partition}_per_sample_correctness']


def get_good_bad_datapoints(partition: str, flags: list[bool]):
    data = load_dataset(config.MATHQA_DIR, partition)
    assert len(data) == len(flags)
    good = []
    bad = []

    for sample, correct in zip(data, flags):
        if correct:
            good.append(sample)
        else:
            bad.append(sample)
    return good, bad


# print the average length, min and maximum length
def get_program_lengths(mathqa_entries: list[RawMathQAEntry]) -> list[int]:
    lengths = []
    for entry in mathqa_entries:
        lengths.append(len(entry.linear_formula))
    return lengths


def get_min_max_mean(num_list: list[int]) -> tuple[int, int, float]:
    return min(num_list), max(num_list), mean(num_list)


def get_stats(datapoints: list[RawMathQAEntry]):
    lengths = get_program_lengths(datapoints)
    return get_min_max_mean(lengths)


def print_stats():
    macro_10_correctness = per_sample_correctness('macro_10', 'train')
    vanilla_correctness = per_sample_correctness('vanilla', 'train')
    macro_10_good, macro_10_bad = get_good_bad_datapoints('train', macro_10_correctness)
    vanilla_good, vanilla_bad = get_good_bad_datapoints('train', vanilla_correctness)
    print(get_stats(vanilla_bad))
    print(get_stats(macro_10_bad))
    print(get_stats(vanilla_good))
    print(get_stats(macro_10_good))


print_stats()