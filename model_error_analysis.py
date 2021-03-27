from statistics import mean
from typing import Union
import itertools

import torch
import numpy as np

from config import get_experiment_logs
from math_qa.math_qa import load_dataset, RawMathQAEntry
from train_mathqa import evaluate_datapoint
from simple_seq2seq import Seq2Seq
from mathqa_processing import MathQAManager, ErrorType
import config
from pathlib import Path

# TODO: categorize the errors to syntax errors and value errors
# TODO: find the 4 sets - both correct, both right, macro correct and vanilla incorrect, and vice versa
# TODO: iterate through a debugger through one of those errors
# TODO: analyze the results with the more epochs
# TODO: make this also used when saving the model

_logger = config.get_logger(__file__)


def get_model_path(experiment_name: str) -> Path:
    return config.TRAINING_LOGS_DIR / experiment_name / 'model.pt'


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


# TODO: make a get_manager and get_model functions.
# TODO: save the special configurations for an experiment in a file to be automatically reproduced
def get_error_types(device, logs: dict, manager: MathQAManager, model, partition: str) -> list[ErrorType]:
    per_sample_correctness = logs[f'{partition}_per_sample_correctness']
    # go find out, out of all the dev errors for the macro_10 model,
    # what percentage of them are syntax errors
    per_sample_error_type = []
    samples = manager.iter_dataset(partition)
    for i, (correct, datapoint) in enumerate(zip(per_sample_correctness, samples)):
        if not correct:
            error_type = evaluate_datapoint(model, manager, datapoint, device, return_error_type=True)
            per_sample_error_type.append(error_type)
        else:
            per_sample_error_type.append(ErrorType.no_error)

    return per_sample_error_type


def count_errors(error_type_list: list[ErrorType], return_percent=False) -> tuple:
    n_correct = 0
    n_value = 0
    n_syntax = 0
    for error_type in error_type_list:
        if error_type == ErrorType.no_error:
            n_correct += 1
        elif error_type == ErrorType.syntax:
            n_syntax += 1
        elif error_type == ErrorType.value:
            n_value += 1
        else:
            assert False, "should not get here"
    if return_percent:
        total = n_value + n_correct + n_syntax
        return n_correct / total, n_syntax / total, n_value / total
    return n_correct, n_syntax, n_value


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load the mathqa manager
    vanilla_manager = MathQAManager(
        config.MATHQA_DIR,
        config.MAX_VOCABULARY_SIZE,
    )
    macro_10_manager = MathQAManager(
        config.MATHQA_DIR,
        config.MAX_VOCABULARY_SIZE,
        macro_file=config.MACRO_10_FILE
    )
    # load the training logs
    vanilla_logs = get_experiment_logs('vanilla')
    macro_10_logs = get_experiment_logs('macro_10')
    # load the models
    macro_10_model = Seq2Seq(
        macro_10_manager.text_vocabulary_size,
        macro_10_manager.code_vocabulary_size,
        config.INTERNAL_DIM,
    ).to(device)
    state_dict_path = get_model_path('macro_10')
    state_dict = torch.load(state_dict_path, map_location=device)
    macro_10_model.load_state_dict(state_dict)
    vanilla_model = Seq2Seq(
        vanilla_manager.text_vocabulary_size,
        vanilla_manager.code_vocabulary_size,
        config.INTERNAL_DIM
    ).to(device)
    state_dict_path = get_model_path('vanilla')
    state_dict = torch.load(state_dict_path, map_location=device)
    vanilla_model.load_state_dict(state_dict)
    # collect all the errors
    partition = 'test'
    macro_10_error_type = get_error_types(device, macro_10_logs, macro_10_manager, macro_10_model, partition)
    vanilla_error_type = get_error_types(device, vanilla_logs, vanilla_manager, vanilla_model, partition)
    # errors = count_errors(macro_10_error_type, return_percent=True)
    # print(f"macro: correct={errors[0]:.2f}, syntax={errors[1]:.2f}, value={errors[2]:.2f}")
    # errors = count_errors(vanilla_error_type, return_percent=True)
    # print(f"vanilla: correct={errors[0]:.2f}, syntax={errors[1]:.2f}, value={errors[2]:.2f}")
    # sample 20 samples from both correct, both wrong, one wrong one correct
    n_samples = 20
    macro_10_correct = np.array([et == ErrorType.no_error for et in macro_10_error_type])
    vanilla_correct = np.array([et == ErrorType.no_error for et in vanilla_error_type])
    indices = np.arange(len(macro_10_correct))
    for macro_10_flag, vanilla_flag in itertools.product([True, False], repeat=2):
        mask = (vanilla_correct == vanilla_flag) & (macro_10_correct == macro_10_flag)
        mask = mask / mask.sum()
        sample_indices = np.random.choice(indices, n_samples, replace=False, p=mask)
        print(sample_indices)


if __name__ == "__main__":
    main()
