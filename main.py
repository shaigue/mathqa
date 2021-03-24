"""Main script to run on the server"""
import os
from pathlib import Path

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
    prefix = config.TRAINING_LOGS_DIR / 'macro'
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
    train(prefix, model, manager, n_epochs=N_EPOCHS)


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
    train(prefix, model, manager, n_epochs=N_EPOCHS)


if __name__ == "__main__":
    train_mathqa_vanilla_dropout()
    train_mathqa_macro_dropout()
