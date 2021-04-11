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
TRAINING_LOGS_DIR = ROOT_DIR / 'training_logs'
LOGS_DIR = ROOT_DIR / 'logs'

# ============================ Naming conventions ==============================
# TODO: move here other naming conventions in the code
# TODO: enable checkpoint loading if training was interrupted in the middle
# TODO: add word vectors to the system
# TODO: try to build a encoder - decoder with copy mechanism + works by considering the relative inputs
#  + the hidden states of past created node.
# TODO: save the special configurations for an experiment in a file to be automatically reproduced


# ========================== For accessing the macro files ===================================
def get_macro_file(n_macros: int) -> Path:
    if n_macros == 0:
        return None
    return MACRO_DIR / f"{n_macros}.json"


# ============================ for accessing training logs ====================================

def get_exp_dir_path(exp_name: str) -> Path:
    return TRAINING_LOGS_DIR / exp_name


def get_exp_model_path(exp_name: str) -> Path:
    return get_exp_dir_path(exp_name) / 'model.pt'


def get_exp_train_log_path(exp_name: str) -> Path:
    return get_exp_dir_path(exp_name) / 'train_log.json'


def load_exp_train_log(exp_name: str) -> dict:
    json_file = get_exp_train_log_path(exp_name)
    with json_file.open('r') as f:
        return json.load(f)


# ============================ This is for logging ============================================

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


def get_old_correctness_rate(train_log: dict, part: str) -> float:
    return train_log[f'{part}_correctness_rate'][-1]['value']


def get_new_correctness_rate(train_log: dict, part: str) -> float:
    d = train_log[part]['correctness_rate']
    k, v = d.popitem()
    d[k] = v
    return v