from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
MATHQA_DIR = DATA_DIR / 'MathQA'

MAX_VOCABULARY_SIZE = 10_000
INTERNAL_DIM = 128

# TODO: enable checkpoint loading if trainig was interapted in the middle