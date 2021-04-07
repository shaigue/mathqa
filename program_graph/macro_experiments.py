"""This is experiments to get intuitions about macros extraction"""

from program_graph.macro_substitution import perform_macro_augmentation_on_train
from program_graph.program import Program


def extract_many_macros():
    """extract 100 macros and run training on 20, 40, 60, 80, 100"""
    perform_macro_augmentation_on_train(100, save_every=20)


def substitute_macros(macros: dict[str, Program], programs: list[Program]) -> list[Program]:
    """Takes a list of of programs, and a dictionary with 'name': macro and substitutes
    the given name from the programs that have those macros.
    """
    # TODO:
    pass



# TODO: add data augmentation during training by permuting the sequence
# TODO: add constants to macros
# TODO: add the ability to prioritize larger macros
# TODO: filter short programs from the dataset, and check what happens
# TODO: filter programs with small amounts of operations types in them
# TODO: add fixed vocabulary
# TODO: add RL
# TODO: add state machine decode


