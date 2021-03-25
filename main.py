"""Main script to run on the server"""
import os
from pathlib import Path

from program_graph.macro_substitution import perform_macro_augmentation_on_train
import config
from mathqa_processing import MathQAManager
from teacher_forcing_gru_encoder_decoder import Seq2Seq
from train_mathqa import train, evaluate


def try_gpu():
    print("STARTING SCRIPT...")
    try:
        import torch
        print("import torch SUCCESS")
        if torch.cuda.is_available():
            print("having gpu!!!")
        else:
            print("not having gpu")

    except:
        print("import torch FAIL")
    print("FINISHED")


N_EPOCHS = 100


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
                            macro_file=config.MACRO_DATA_FILE)
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
    prefix = config.TRAINING_LOGS_DIR / 'vanilla_dropout'
    prefix.mkdir(exist_ok=True)
    train(prefix, model, manager, n_epochs=N_EPOCHS+50)


def train_mathqa_macro_dropout():
    manager = MathQAManager(root_dir=config.MATHQA_DIR, max_vocabulary_size=config.MAX_VOCABULARY_SIZE, dummy=False,
                            macro_file=config.MACRO_DATA_FILE)
    model = Seq2Seq(
        source_vocabulary_size=manager.text_vocabulary_size,
        target_vocabulary_size=manager.code_vocabulary_size,
        hidden_dim=config.INTERNAL_DIM,
        dropout=0.2
    )
    prefix = config.TRAINING_LOGS_DIR / 'macro_dropout'
    prefix.mkdir(exist_ok=True)
    train(prefix, model, manager, n_epochs=N_EPOCHS+50)


def different_number_of_macros():
    print("starting to extract macros...", flush=True)
    perform_macro_augmentation_on_train(9, save_every=2)

    for n_macros in range(1, 10, 2):
        print(f"starting training with n_macros={n_macros}...", flush=True)
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
    different_number_of_macros()
    # train_mathqa_vanilla_dropout()
    # train_mathqa_macro_dropout()
