"""Main script to run on the server"""
import config
from mathqa_processing import MathQAManager
from simple_seq2seq import Seq2Seq
from train_mathqa import train, get_manager, get_model

_logger = config.get_logger(__file__)
N_EPOCHS = 100


def run_dropout_2_gru_layer_experiment():
    n_epochs = 300
    for n_macros in [0, 10]:
        macro_file = config.get_n_macro_file(n_macros)
        manager = get_manager(macro_file=macro_file)
        for dropout, layers in [(0.2, 1), (0, 2), (0.2, 2)]:
            model = get_model(manager, dropout, layers)
            exp_name = f"macro_{n_macros}_dropout_{dropout}_layers_{layers}"
            exp_dir = config.get_experiment_dir_path(exp_name)
            train(dir_path=exp_dir, model=model, manager=manager, n_epochs=n_epochs)


def run_multiple_macro_experiments(num_macros: int, prefix='', n_epochs=200):
    macro_file = config.get_n_macro_file(num_macros)
    manager = get_manager(macro_file=macro_file)
    model = get_model(manager)
    prefix = config.get_macro_prefix(num_macros, prefix)
    train(dir_path=prefix, model=model, manager=manager, n_epochs=n_epochs, evaluate_every=10)


def run_all_no_punc_experiments():
    for n_macros in [0, 1, 5, 10]:
        run_multiple_macro_experiments(n_macros, 'no_punc_')


def run_all_no_punc_experiments_converge():
    for n_macros in [0, 5, 10]:
        run_multiple_macro_experiments(n_macros, 'converge_', n_epochs=800)


def train_mathqa_vanilla():
    manager = MathQAManager(root_dir=config.MATHQA_DIR, max_vocabulary_size=config.MAX_VOCABULARY_SIZE, dummy=False)
    model = Seq2Seq(
        source_vocabulary_size=manager.text_vocabulary_size,
        target_vocabulary_size=manager.code_vocabulary_size,
        hidden_dim=config.INTERNAL_DIM
    )
    prefix = config.TRAINING_LOGS_DIR / 'vanilla'
    prefix.mkdir(exist_ok=True)
    train(prefix, model, manager, n_epochs=N_EPOCHS)


def train_mathqa_macro():
    manager = MathQAManager(root_dir=config.MATHQA_DIR, max_vocabulary_size=config.MAX_VOCABULARY_SIZE, dummy=False,
                            macro_file=config.MACRO_10_FILE)
    model = Seq2Seq(
        source_vocabulary_size=manager.text_vocabulary_size,
        target_vocabulary_size=manager.code_vocabulary_size,
        hidden_dim=config.INTERNAL_DIM
    )
    prefix = config.TRAINING_LOGS_DIR / 'macro_10'
    prefix.mkdir(exist_ok=True)
    train(prefix, model, manager, n_epochs=N_EPOCHS)


def train_mathqa_vanilla_dropout():
    manager = MathQAManager(root_dir=config.MATHQA_DIR, max_vocabulary_size=config.MAX_VOCABULARY_SIZE, dummy=False)
    model = Seq2Seq(
        source_vocabulary_size=manager.text_vocabulary_size,
        target_vocabulary_size=manager.code_vocabulary_size,
        hidden_dim=config.INTERNAL_DIM,
        dropout=0.2
    )
    prefix = config.TRAINING_LOGS_DIR / 'vanilla_150'
    prefix.mkdir(exist_ok=True)
    train(prefix, model, manager, n_epochs=N_EPOCHS+50)


def train_mathqa_macro_dropout():
    manager = MathQAManager(root_dir=config.MATHQA_DIR, max_vocabulary_size=config.MAX_VOCABULARY_SIZE, dummy=False,
                            macro_file=config.MACRO_10_FILE)
    model = Seq2Seq(
        source_vocabulary_size=manager.text_vocabulary_size,
        target_vocabulary_size=manager.code_vocabulary_size,
        hidden_dim=config.INTERNAL_DIM,
        dropout=0.2
    )
    prefix = config.TRAINING_LOGS_DIR / 'macro_10_150'
    prefix.mkdir(exist_ok=True)
    train(prefix, model, manager, n_epochs=N_EPOCHS+50)


def different_number_of_macros():

    for n_macros in range(1, 10, 2):
        macro_file = config.MACRO_DIR / f'{n_macros}.pkl'
        manager = MathQAManager(root_dir=config.MATHQA_DIR, max_vocabulary_size=config.MAX_VOCABULARY_SIZE, dummy=False,
                                macro_file=macro_file)
        model = Seq2Seq(
            source_vocabulary_size=manager.text_vocabulary_size,
            target_vocabulary_size=manager.code_vocabulary_size,
            hidden_dim=config.INTERNAL_DIM,
        )
        prefix = config.TRAINING_LOGS_DIR / f'macro_{n_macros}'
        prefix.mkdir(exist_ok=True)
        train(prefix, model, manager, n_epochs=N_EPOCHS)


if __name__ == "__main__":
    run_dropout_2_gru_layer_experiment()
