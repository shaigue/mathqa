from pathlib import Path

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
MATHQA_DIR = DATA_DIR / 'MathQA'

MAX_VOCABULARY_SIZE = 10_000
INTERNAL_DIM = 128

MIN_MACRO_SIZE = 2
MAX_MACRO_SIZE = 8
MAX_MACRO_INPUTS = 5

MACRO_DIR = ROOT_DIR / 'macros'
MACRO_DATA_FILE = MACRO_DIR / 'macro_10.pkl'
TRAINING_LOGS_DIR = ROOT_DIR / 'training_logs'


# TODO: enable checkpoint loading if training was interrupted in the middle
