"""This is a script to augment the dataset with macros, by replacing them in the real dataset"""
import json
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import config
from math_qa import math_qa
from math_qa.math_qa import RawMathQAEntry
from program_graph.program import Program, OperationNode


# setting the logger for this module
_logger = config.get_logger(__file__)


def extract_programs(entries: list[RawMathQAEntry]) -> list[Program]:
    programs = []
    for datapoint in entries:
        program = Program.from_linear_formula(datapoint.linear_formula)
        programs.append(program)
    return programs


def get_programs(partition: str) -> list[Program]:
    """

    :param partition:
    :return: a list of all the programs in the partition, with the same ordering
    """
    data = math_qa.load_dataset(config.MATHQA_DIR, partition)
    return extract_programs(data)


@dataclass(frozen=True)
class MacroAssociation:
    index: int
    vertex_subset: frozenset[OperationNode]

    def conflicts(self, other) -> bool:
        """Checks if 2 macro_10 associations conflict, i.e. whether they have the same index and non-empty intersection
        in
        in their vertex_subset.

        :param other:
        :return:
        """
        return self.index == other.index and len(self.vertex_subset.intersection(other.vertex_subset)) > 0


def get_macros_with_association(program: Program, program_index: int,
                                min_macro_size: int, max_macro_size: int,
                                max_macro_inputs: int) -> dict[Program, list[MacroAssociation]]:
    """Finds for a single program all the macros that are associated with it,
    and logs the vertex indices that each macro_10 involves.

    :param program:
    :param program_index:
    :return:
    """
    macro_iterator = program.function_cut_iterator(
        min_size=min_macro_size,
        max_size=max_macro_size,
        max_inputs=max_macro_inputs,
        return_subsets=True
    )
    macro_dict = defaultdict(list)
    for macro, vertex_subset in macro_iterator:
        macro_dict[macro].append(MacroAssociation(program_index, vertex_subset))
    return dict(macro_dict)


def get_all_macro_associations(program_list: list[Program], min_macro_size: int,
                               max_macro_size: int, max_macro_inputs: int,
                               ) -> dict[Program, list[MacroAssociation]]:
    """Collects all the macros and their associations in the list of programs

    :param program_list:
    :return:
    """
    macro_accumulator = defaultdict(list)
    for program_index, program in enumerate(program_list):
        if (program_index + 1) % 100 == 0:
            _logger.info(f"extracting macros from program={program_index + 1} out of {len(program_list)}")
        current_program_macro_dict = get_macros_with_association(
            program,
            program_index,
            min_macro_size,
            max_macro_size,
            max_macro_inputs
        )
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
    """Copies the current macro_10 associations, and removes from it all the associations that conflict with the
    macro_10
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
    """Given macro_10 associations, selects the one with the highest score."""
    assert len(macro_associations) > 0, "got empty macro_associations"
    max_macro = None
    max_score = 0
    for macro, associations in macro_associations.items():
        score = macro_score(macro, len(associations))
        if max_score < score:
            max_macro = macro
            max_score = score

    assert max_macro is not None, "trying to return a None macro_10."
    return max_macro


def get_macro_symbol(macro_num: int) -> str:
    return f"macro_{macro_num}"


def replace_macro_in_list(program_list: list[Program],
                          macro_num: int, macro: Program, associations: list[MacroAssociation]) -> list[Program]:
    """Returns a modified list where the macro_10 has been substituted in the associated parts"""
    symbol = get_macro_symbol(macro_num)
    transformed_programs = deepcopy(program_list)
    for association in associations:
        program_index = association.index
        transformed_programs[program_index] = transformed_programs[program_index].refactor_macro(
            association.vertex_subset, macro, symbol)

    return transformed_programs


def save_macro_extraction(extracted_macros: list[Program], modified_programs: list[Program], file: Path):
    extracted_macros = [str(macro) for macro in extracted_macros]
    modified_programs = [str(program) for program in modified_programs]
    data = {'extracted_macros': extracted_macros, 'modified_programs': modified_programs}
    with file.open('w') as f:
        json.dump(data, f)


def load_macro_extraction(file: Path) -> tuple[list[Program], list[Program]]:
    """The output is first the list macros"""
    with open(file, 'r') as f:
        data = json.load(f)

    extracted_macros = [Program.from_linear_formula(macro) for macro in data['extracted_macros']]
    modified_programs = [Program.from_linear_formula(program) for program in data['modified_programs']]
    return extracted_macros, modified_programs


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
    """Removes for each macro_10 any conflicting associations"""
    new_associations = {}
    for macro, associations in macro_associations.items():
        new_associations[macro] = filter_conflicting_associations(associations)
    return new_associations


def perform_macro_augmentation_on_train(n_macros: int, save_every=None,
                                        min_macro_size=config.MIN_MACRO_SIZE,
                                        max_macro_size=config.MAX_MACRO_SIZE,
                                        max_macro_inputs=config.MAX_MACRO_INPUTS,
                                        data: list[RawMathQAEntry] = None,
                                        target_file: Path = None):
    if data is None:
        program_list = get_programs('train')
    else:
        program_list = extract_programs(data)
    # extract all the macro_10 associations
    macro_associations = get_all_macro_associations(program_list, min_macro_size,
                                                    max_macro_size, max_macro_inputs)
    # removing the self conflicts from the list
    macro_associations = filter_self_conflicting_macros(macro_associations)
    # extract n_macros
    extracted_macros = []
    for macro_num in range(n_macros):
        _logger.info(f"substituting macro={macro_num + 1} out of {n_macros}")
        # find the best ranking one
        best_macro = find_best_macro(macro_associations)
        # replace all it's appearances in the programs
        program_list = replace_macro_in_list(program_list, macro_num, best_macro, macro_associations[best_macro])
        # remove the conflicting macros
        macro_associations = remove_conflicting_associations(macro_associations, best_macro)
        # remove the extracted macro_10
        macro_associations.pop(best_macro)
        extracted_macros.append(best_macro)

        if save_every is not None and (macro_num % save_every) == 0 and macro_num != (n_macros - 1):
            _logger.info(f"saving macro file no. {macro_num + 1}")
            save_macro_extraction(extracted_macros, program_list, config.get_macro_file(n_macros))
    # save the required data to a file
    _logger.info(f"finished extracting macros, saving to file")
    if target_file is None:
        save_macro_extraction(extracted_macros, program_list, config.get_macro_file(n_macros))
    else:
        save_macro_extraction(extracted_macros, program_list, target_file)


def example():
    perform_macro_augmentation_on_train(9, save_every=2)


if __name__ == "__main__":
    example()
