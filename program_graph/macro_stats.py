import logging
import pickle
import time
from collections import Counter

import pandas as pd

import config
from math_qa.dataset import _load_json
from program_graph.extract_dags import Program


def get_data(part: str):
    return _load_json(config.MATHQA_DIR / f'{part}.json')


def count_repeating_functions(part, min_size, max_size, max_inputs) -> Counter:
    pickle_file = f"{part}_function_counter.pkl"
    function_counter = Counter()
    data = get_data(part)
    for i, dp in enumerate(data):
        logging.info(f"{i+1} out of {len(data)} in {part}...")
        program = Program.from_linear_formula(dp.linear_formula)
        function_counter.update(program.function_cut_iterator(min_size, max_size, max_inputs))

    with open(pickle_file, "wb") as fd:
        pickle.dump(function_counter, fd)

    return function_counter


def total_count(counter: Counter) -> int:
    count = 0
    for key, amount in counter.items():
        count += amount
    return count


def count_for_each_partition(min_size, max_size, max_inputs):
    partitions = ['test', 'train', 'dev']
    rows_list = []
    counters = {part: count_repeating_functions(part, min_size, max_size, max_inputs) for part in partitions}
    programs_per_part = {part: total_count(counters[part]) for part in partitions}
    for part in partitions:
        counter = counters[part]
        part_programs = list(counter.keys())
        for program in part_programs:
            row = {part0: counters[part0].pop(program, 0) for part0 in partitions}
            row['linear_formula'] = str(program)
            row['size'] = program.get_n_operations()
            row['n_inputs'] = program.get_n_inputs()
            rows_list.append(row)
    data_frame = pd.DataFrame(rows_list, columns=['linear_formula', 'size', 'n_inputs'] + partitions)
    data_frame.sort_values(by=partitions, ascending=False, inplace=True, ignore_index=True)
    # data_frame = data_frame[:10_000]
    # save all entries
    data_frame.to_csv('count_for_each_partition.csv', index=False)
    # add another entry where the points are normalized with their relative percentage
    for part in partitions:
        data_frame[part] /= programs_per_part[part]
    data_frame.to_csv('percent_for_each_partition.csv', index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filemode='w', filename='log.txt')
    start_time = time.time()
    count_for_each_partition(min_size=2, max_size=10, max_inputs=5)
    elapsed_time = time.time() - start_time
    logging.info(f"***finished***")
    logging.info(f"time={elapsed_time} seconds")
