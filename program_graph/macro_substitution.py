"""This is a script to augment the dataset with macros, by replacing them in the real dataset"""
from collections import defaultdict
from dataclasses import dataclass
from copy import deepcopy
import pickle
from pathlib import Path

import math_qa.dataset as mathqa
from program_graph.extract_dags import Program, Node
import config
import logging
import time


def get_programs(partition: str) -> list[Program]:
    """

    :param partition:
    :return: a list of all the programs in the partition, with the same ordering
    """
    data = mathqa.load_dataset(config.MATHQA_DIR, partition)
    programs = []
    for datapoint in data:
        program = Program.from_linear_formula(datapoint.linear_formula)
        programs.append(program)
    return programs


@dataclass(frozen=True)
class MacroAssociation:
    index: int
    vertex_subset: frozenset[Node]

    def conflicts(self, other) -> bool:
        """Checks if 2 macro associations conflict, i.e. whether they have the same index and non-empty intersection in
        in their vertex_subset.

        :param other:
        :return:
        """
        if __name__ == '__main__':
            return self.index == other.index and len(self.vertex_subset.intersection(other.vertex_subset)) > 0


def get_macros_with_association(program: Program, program_index: int) -> dict[Program, list[MacroAssociation]]:
    """Finds for a single program all the macros that are associated with it,
    and logs the vertex indices that each macro involves.

    :param program:
    :param program_index:
    :return:
    """
    macro_iterator = program.function_cut_iterator(
        min_size=config.MIN_MACRO_SIZE,
        max_size=config.MAX_MACRO_SIZE,
        max_inputs=config.MAX_MACRO_INPUTS,
        return_subsets=True
    )
    macro_dict = defaultdict(list)
    for macro, vertex_subset in macro_iterator:
        macro_dict[macro].append(MacroAssociation(program_index, vertex_subset))
    return dict(macro_dict)


def get_all_macro_associations(program_list: list[Program]) -> dict[Program, list[MacroAssociation]]:
    """Collects all the macros and their associations in the list of programs

    :param program_list:
    :return:
    """
    macro_accumulator = defaultdict(list)
    for program_index, program in enumerate(program_list):
        if (program_index + 1) % 100 == 0:
            logging.info(f"extracting macro from program number={program_index + 1}")
        current_program_macro_dict = get_macros_with_association(program, program_index)
        # this will append the lists in the current locations
        for macro, association_list in current_program_macro_dict.items():
            macro_accumulator[macro] += association_list
    return dict(macro_accumulator)


def remove_conflicts_from_list(target_list: list[MacroAssociation], to_remove_list: list[MacroAssociation]):
    """returns a list that contains only the associations that do not conflict with any in the target_list"""
    associations_to_keep = []
    for association in to_remove_list:

        conflicts = False
        for target_association in target_list:
            if association.conflicts(target_association):
                conflicts = True
                break
        if not conflicts:
            associations_to_keep.append(association)
    return associations_to_keep


def remove_conflicting_associations(macro_associations: dict[Program, list[MacroAssociation]],
                                    target_macro: Program) -> dict[Program, list[MacroAssociation]]:
    """Copies the current macro associations, and removes from it all the associations that conflict with the macro
    """
    assert target_macro in macro_associations, "target_macro has to be in macro_associations."

    non_conflicting_associations = {}
    target_macro_associations = macro_associations[target_macro]
    non_conflicting_associations[target_macro] = target_macro_associations

    for macro, associations in macro_associations.items():
        if macro != target_macro:
            non_conflicting_associations[macro] = remove_conflicts_from_list(target_macro_associations, associations)

    return non_conflicting_associations


def macro_score(program: Program, times_appeared: int) -> float:
    """The score is N * (3 * S - I) where:
    N - number of times appeared
    S - number of operators in it
    I - number of inputs to it
    this is for the amount of symbols that we will save
    """
    n_inputs = program.get_n_inputs()
    n_operations = program.get_n_operations()
    return times_appeared * (3 * n_operations - n_inputs)


def find_best_macro(macro_associations: dict[Program, list[MacroAssociation]]) -> Program:
    """Given macro associations, selects the one with the highest score."""
    assert len(macro_associations) > 0, "got empty macro_associations"
    max_macro = None
    max_score = 0
    for macro, associations in macro_associations.items():
        score = macro_score(macro, len(associations))
        if max_score < score:
            max_macro = macro
            max_score = score

    assert max_macro is not None, "trying to return a None macro."
    return max_macro


def get_macro_symbol(macro_num: int) -> str:
    return f"macro_{macro_num}"


def replace_macro_in_list(program_list: list[Program],
                          macro_num: int, macro: Program, associations: list[MacroAssociation]) -> list[Program]:
    """Returns a modified list where the macro has been substituted in the associated parts"""
    symbol = get_macro_symbol(macro_num)
    transformed_programs = deepcopy(program_list)
    for association in associations:
        program_index = association.index
        transformed_programs[program_index] = transformed_programs[program_index].refactor_macro(
            association.vertex_subset, macro, symbol)

    return transformed_programs


def save_macro_extraction(extracted_macros: list[Program], modified_programs: list[Program], file: Path):
    data = {'extracted_macros': extracted_macros, 'modified_programs': modified_programs}
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def load_macro_extraction(file: Path) -> tuple[list[Program], list[Program]]:
    """The output is first the list macros"""
    with open(file, 'rb') as f:
        data = pickle.load(f)

    return data['extracted_macros'], data['modified_programs']


class MacroData:
    def __init__(self, macro_dict: dict[str, Program], modified_programs: list[Program]):
        self.macro_dict = macro_dict
        self.modified_programs = modified_programs

    @classmethod
    def from_file(cls, file: Path):
        extracted_macros, modified_programs = load_macro_extraction(file)
        macro_dict = {get_macro_symbol(i): macro for i, macro in enumerate(extracted_macros)}
        return cls(macro_dict, modified_programs)


def filter_conflicting_associations(associations: list[MacroAssociation]) -> list[MacroAssociation]:
    new_associations = []
    n_associations = len(associations)
    conflict_found = [False] * n_associations
    for i in range(n_associations):
        if conflict_found[i]:
            continue
        current_association = associations[i]
        new_associations.append(current_association)
        # update the future conflict
        for j in range(i + 1, n_associations):
            if associations[j].conflicts(current_association):
                conflict_found[j] = True
    return new_associations


def filter_self_conflicting_macros(macro_associations: dict[Program, list[MacroAssociation]]) -> \
        dict[Program, list[MacroAssociation]]:
    """Removes for each macro any conflicting associations"""
    new_associations = {}
    for macro, associations in macro_associations.items():
        new_associations[macro] = filter_conflicting_associations(associations)
    return new_associations


def perform_macro_augmentation_on_train(n_macros: int):
    program_list = get_programs('train')
    # extract all the macro associations
    logging.info(f"getting all macro associations...")
    macro_associations = get_all_macro_associations(program_list)
    # removing the self conflicts from the list
    macro_associations = filter_self_conflicting_macros(macro_associations)
    # extract n_macros
    extracted_macros = []
    for macro_num in range(n_macros):
        logging.info(f"replacing macro {macro_num}")
        # find the best ranking one
        best_macro = find_best_macro(macro_associations)
        # replace all it's appearances in the programs
        program_list = replace_macro_in_list(program_list, macro_num, best_macro, macro_associations[best_macro])
        # remove the conflicting macros
        macro_associations = remove_conflicting_associations(macro_associations, best_macro)
        # remove the extracted macro
        macro_associations.pop(best_macro)
        extracted_macros.append(best_macro)
    # save the required data to a file
    logging.info("finished refactoring, saving macros and programs to file...")
    save_macro_extraction(extracted_macros, program_list)


def example():
    logging.basicConfig(filename='macro_extraction_logs.log', filemode='w', level=logging.INFO)
    start_time = time.time()
    logging.info("starting to log...")
    perform_macro_augmentation_on_train(10)
    end_time = time.time()
    logging.info(f"finished. total time={end_time - start_time}")


if __name__ == "__main__":
    example()
