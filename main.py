"""Main script to run on the server"""
import config
from mathqa_processing import MathQAManager
from teacher_forcing_gru_encoder_decoder import Seq2Seq
from train_seq2seq_teacher_forcing import train, evaluate

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


def train_mathqa():
    manager = MathQAManager(root_dir=config.MATHQA_DIR, max_vocabulary_size=config.MAX_VOCABULARY_SIZE, dummy=False)
    model = Seq2Seq(
        source_vocabulary_size=manager.text_vocabulary_size,
        target_vocabulary_size=manager.code_vocabulary_size,
        hidden_dim=config.INTERNAL_DIM
    )
    train(model, manager, n_epochs=100)


if __name__ == "__main__":
    train_mathqa()
