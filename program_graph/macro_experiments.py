"""This is experiments to get intuitions about macros extraction"""
from fractions import Fraction

import config
from math_qa.math_qa import load_all_dataset, RawMathQAEntry
from program_graph.macro_substitution import perform_macro_augmentation_on_train
from program_graph.program import Program
from train_mathqa import get_manager, get_model, train

_logger = config.get_logger(__file__)

def extract_many_macros():
    """extract 100 macros and run training on 20, 40, 60, 80, 100"""
    perform_macro_augmentation_on_train(100, save_every=20)


def substitute_macros(macros: dict[str, Program], programs: list[Program]) -> list[Program]:
    """Takes a list of of programs, and a dictionary with 'name': macro and substitutes
    the given name from the programs that have those macros.
    """
    # TODO:
    pass


def select_samples(samples_lens: list[int], n_samples: int, target_avg: int) -> list[int]:
    # start by sampling all the samples that have the target length
    selected_indices = []
    for i, n in enumerate(samples_lens):
        if n == target_avg:
            selected_indices.append(i)
    # if there is enough samples then return
    if len(selected_indices) >= n_samples:
        return selected_indices[:n_samples]
    # continue until we have enough samples
    top_diff, bottom_diff = 1, 1
    while len(selected_indices) < n_samples:
        # set the bottom and top sets
        bottom, top = [], []
        for i, n in enumerate(samples_lens):
            if i in selected_indices:
                continue
            elif n == target_avg + top_diff:
                top.append(i)
            elif n == target_avg - bottom_diff:
                bottom.append(i)
        # find the sample ratio between top and bottom
        bottom_ratio, top_ratio = Fraction(top_diff, bottom_diff).as_integer_ratio()
        # as long as there are enough samples to keep things balanced:
        while len(bottom) >= bottom_ratio and len(top) >= top_ratio and len(selected_indices) < n_samples:
            for _ in range(bottom_ratio):
                selected_indices.append(bottom.pop(0))
            for _ in range(top_ratio):
                selected_indices.append(top.pop(0))
        # increment by one the differences
        if len(bottom) < bottom_ratio:
            bottom_diff += 1
        elif len(top) < top_ratio:
            top_diff += 1

    return selected_indices


def get_n_ops(raw_entry: RawMathQAEntry) -> int:
    linear_formula = raw_entry.linear_formula
    if linear_formula[-1] == '|':
        linear_formula = linear_formula[:-1]
    n_ops = linear_formula.count('|')
    return n_ops


def get_n_samples(n_total_samples: int, frac: float) -> int:
    return round(n_total_samples * frac)


def get_subset_with_avg_ops(data: dict[str, list[RawMathQAEntry]],
                            avg_len: int, data_frac: float) -> dict[str, list[RawMathQAEntry]]:
    subset_data = {}
    for part, entries in data.items():
        n_total_samples = len(entries)
        n_samples = get_n_samples(n_total_samples, data_frac)
        n_ops = [get_n_ops(entry) for entry in entries]
        subset_indices = select_samples(n_ops, n_samples, avg_len)
        # save the subsets into the logs - to check that it makes sense
        print(f"subset indices len={len(subset_indices)} in {part}, avg_len={avg_len}")
        subset_data[part] = [entries[i] for i in subset_indices]

    return subset_data


def different_avg_len_macros():
    all_data = load_all_dataset()
    data_frac = 1 / 3
    # lens = [2, 3, 4, 5, 6]
    lens = [5, 6]
    for avg_len in lens:
        _logger.info(f'starting {avg_len}')
        exp_name = f'diff_avg_len_{avg_len}'
        # get the correct subset of the data
        subset_data = get_subset_with_avg_ops(all_data, avg_len, data_frac)

        if avg_len != 5:  # TODO: hacky
            # load the manager with the regular data
            manager = get_manager(raw_data=subset_data)
            model = get_model(manager)
            exp_dir = config.get_exp_dir_path(exp_name)
            # train the model on it
            _logger.info(f'training no macro')
            train(exp_dir, model, manager, 200, 10)

        # extract 10 macros out of it and save it to a file
        exp_name = f'diff_avg_len_{avg_len}_macro'
        macro_file = config.MACRO_DIR / (exp_name + '.json')
        if avg_len != 5:  # TODO: hacky
            _logger.info(f'extracting macros')
            perform_macro_augmentation_on_train(10, data=subset_data['train'],
                                                target_file=macro_file)
        # load the manager with macros
        manager = get_manager(raw_data=subset_data, macro_file=macro_file)
        model = get_model(manager)
        exp_dir = config.get_exp_dir_path(exp_name)
        # train the model with macros
        _logger.info(f'training with macro')
        train(exp_dir, model, manager, 200, 10)


# TODO: add data augmentation during training by permuting the sequence
# TODO: add constants to macros
# TODO: add the ability to prioritize larger macros
# TODO: filter short programs from the dataset, and check what happens
# TODO: filter programs with small amounts of operations types in them
# TODO: add fixed vocabulary
# TODO: add RL
# TODO: add state machine decode
# TODO: add tensorboard usage in the server


