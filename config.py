import json
from pathlib import Path
import logging

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
MATHQA_DIR = DATA_DIR / 'MathQA'

MAX_VOCABULARY_SIZE = 10_000
INTERNAL_DIM = 128

MIN_MACRO_SIZE = 2
MAX_MACRO_SIZE = 8
MAX_MACRO_INPUTS = 5

MACRO_DIR = ROOT_DIR / 'macros'
MACRO_10_FILE = MACRO_DIR / '10.pkl'
TRAINING_LOGS_DIR = ROOT_DIR / 'training_logs'
LOGS_DIR = ROOT_DIR / 'logs'

# ============================ Naming conventions ==============================
# TODO: move here other naming conventions in the code
# TODO: enable checkpoint loading if training was interrupted in the middle


def get_n_macro_file(n_macros: int) -> Path:
    assert n_macros in [0, 1, 3, 5, 7, 9, 10], "Does not exist"
    if n_macros == 0:
        return None
    return MACRO_DIR / f"{n_macros}.pkl"


def get_no_punc_exp_name(n_macros: int) -> str:
    return f"no_punc_macro_{n_macros}"


def get_no_punc_exp_prefix(n_macros: int) -> Path:
    return TRAINING_LOGS_DIR / get_no_punc_exp_name(n_macros)



def get_log_file(module_name: str):
    return LOGS_DIR / f'{module_name}.log'


def get_logger(file: str, mode='w') -> logging.Logger:
    module_name = Path(file).stem
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(name)s-%(message)s')
    handler = logging.FileHandler(get_log_file(module_name), mode)
    handler.setFormatter(formatter)
    logger = logging.getLogger(module_name)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)  # disable this to drop the logging
    return logger


def get_experiment_logs(experiment_name: str):
    filename = 'train_log.json'
    experiment_dir = TRAINING_LOGS_DIR / experiment_name / filename
    with experiment_dir.open('r') as f:
        experiment_logs = json.load(f)
    return experiment_logs
