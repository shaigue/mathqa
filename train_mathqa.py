"""This is a code to train a sequence to sequence model with teacher forcing"""
from collections import defaultdict
import json
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mathqa_processing import MathQAManager, MathQADatapoint, ErrorReport, ErrorType
from mathqa_torch_loader import get_train_loader
from simple_seq2seq import Seq2Seq
import config

_logger = config.get_logger(__file__, mode='a')


def teacher_forcing_loss(target_token_indices, predicted_target_token_logits, pad_index: int):
    seq_len, batch_size, target_vocabulary_size = predicted_target_token_logits.shape
    assert target_token_indices.shape == (seq_len, batch_size)
    # drop the first target symbol <SOS>
    target_token_indices = target_token_indices[1:]
    # drop the last predicted
    predicted_target_token_logits = predicted_target_token_logits[:-1]
    # swap the axis so that the probabilities over the vocabulary will be in dim=1
    predicted_target_token_logits = torch.transpose(predicted_target_token_logits, 2, 1)
    # output.shape = (N, C, ...), target.shape = (N, ...)
    return F.cross_entropy(predicted_target_token_logits, target_token_indices, ignore_index=pad_index)


def tensorize_token_list(token_list: list[int], device):
    return torch.tensor(token_list, dtype=torch.int64, device=device).view(-1, 1)


def train_batch(model: Seq2Seq, optimizer: torch.optim.Optimizer, batch):
    optimizer.zero_grad()

    source_tokens, target_tokens, source_lens, target_lens = batch
    predicted_target_token_logits = model(source_tokens, target_tokens, source_lens, target_lens)

    loss = teacher_forcing_loss(target_tokens, predicted_target_token_logits, model.pad_index)

    loss.backward()
    optimizer.step()

    return loss.detach().cpu().item()


def train_epoch(model: Seq2Seq, optimizer: torch.optim.Optimizer, train_loader: DataLoader):
    model.train()
    total_loss = 0
    n_batches = 0

    for datapoint_index, batch in enumerate(train_loader):
        batch_loss = train_batch(model, optimizer, batch)
        total_loss += batch_loss
        n_batches += 1

    avg_loss = total_loss / n_batches
    return avg_loss


def evaluate_datapoint(model: Seq2Seq, manager: MathQAManager, datapoint: MathQADatapoint,
                       device) -> ErrorReport:
    source_token_indices = datapoint.text_token_indices
    source_token_indices = tensorize_token_list(source_token_indices, device)
    generated_token_indices = model.generate(
        source_token_indices,
        start_of_string_token_index=manager.code_start_token_index,
        end_of_string_token_index=manager.code_end_token_index,
        max_target_seq_len=manager.code_max_len,
    )
    return manager.check_generated_code_correctness(generated_token_indices, datapoint)


def evaluate(model: Seq2Seq, manager: MathQAManager, partition: str, device,
             return_per_sample=False) -> Union[float, tuple[float, list[ErrorReport]]]:
    model.eval()
    total_correct = 0
    n_batches = 0
    if return_per_sample:
        per_sample = []

    with torch.no_grad():
        for datapoint_index, datapoint in enumerate(manager.iter_dataset(partition, shuffle=False)):
            error_report = evaluate_datapoint(model, manager, datapoint, device)
            correct = error_report.error_type == ErrorType.no_error
            if correct:
                total_correct += 1
            n_batches += 1

            if return_per_sample:
                per_sample.append(error_report)

    correctness_rate = total_correct / n_batches

    if return_per_sample:
        return correctness_rate, per_sample

    return correctness_rate


def convert_error_report_json(er: ErrorReport) -> dict:
    return {'error_type': er.error_type.name, 'generated_tokens': er.generated_tokens}


def train(prefix: Path, model: Seq2Seq, manager: MathQAManager, n_epochs: int = 10, evaluate_every: int = 5,
          logs_file=None, checkpoint_file=None, batch_size=32, lr=0.01, weight_decay=1e-4):
    if logs_file is None:
        logs_file = prefix / 'train_log.json'
    if checkpoint_file is None:
        checkpoint_file = prefix / 'model.pt'

    prefix.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = get_train_loader(manager, device, batch_size)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, verbose=True,
                                                              factor=0.5)

    best_dev_correctness_rate = 0
    logs = defaultdict(list)
    _logger.info(f"using device={device}. starting training session {prefix.name}.")
    for epoch_index in range(n_epochs):
        epoch_loss = train_epoch(model, optimizer, train_loader)
        _logger.info(f"epoch={epoch_index}, loss={epoch_loss}")

        logs['epoch_loss'].append({'epoch': epoch_index, 'value': epoch_loss})

        # checkpoint - save only the best performing model on dev
        if (epoch_index + 1) % evaluate_every == 0 or epoch_index == n_epochs - 1:
            train_correctness_rate = evaluate(model, manager, 'train', device)
            dev_correctness_rate = evaluate(model, manager, 'dev', device)
            _logger.info(f"train_correctness_rate={train_correctness_rate}")
            _logger.info(f"dev_correctness_rate={dev_correctness_rate}")

            logs['train_correctness_rate'].append({'epoch': epoch_index, 'value': train_correctness_rate})
            logs['dev_correctness_rate'].append({'epoch': epoch_index, 'value': dev_correctness_rate})
            # check if to save the model
            if dev_correctness_rate > best_dev_correctness_rate:
                _logger.info(f"saving best model checkpoint at epoch={epoch_index}")
                torch.save(model.state_dict(), checkpoint_file)
                best_dev_correctness_rate = dev_correctness_rate
            # update the scheduler
            lr_scheduler.step(dev_correctness_rate)

    # make sure to load the best performing model on dev
    model.load_state_dict(torch.load(checkpoint_file))
    # evaluate on each one of the partitions, and save their correctness rate and per_sample correctness in the logs
    for part in manager.partitions:
        correctness_rate, per_sample = evaluate(model, manager, part, device, return_per_sample=True)
        _logger.info(f"{part}_correctness_rate={correctness_rate}")
        logs[f'{part}_correctness_rate'].append({'epoch': n_epochs, 'value': correctness_rate})
        logs[f'{part}_per_sample_report'] = list(map(convert_error_report_json, per_sample))
    # save logs
    with open(logs_file, 'w') as f:
        json.dump(logs, f)
    _logger.info(f"{prefix.name} training finished")


def get_manager(no_punctuation: bool = True, macro_file=None, dummy=False, ):
    return MathQAManager(
        root_dir=config.MATHQA_DIR,
        max_vocabulary_size=config.MAX_VOCABULARY_SIZE,
        dummy=dummy,
        macro_file=macro_file,
        no_punctuation=no_punctuation,
    )


def get_model(manager: MathQAManager):
    return Seq2Seq(
        source_vocabulary_size=manager.text_vocabulary_size,
        target_vocabulary_size=manager.code_vocabulary_size,
        hidden_dim=config.INTERNAL_DIM,
        pad_index=manager.pad_index,
        dropout=0,
    )


def simple_example():
    manager = MathQAManager(root_dir=config.MATHQA_DIR, max_vocabulary_size=1000, dummy=True)
    model = Seq2Seq(
        source_vocabulary_size=manager.text_vocabulary_size,
        target_vocabulary_size=manager.code_vocabulary_size,
        hidden_dim=32,
        pad_index=manager.pad_index
    )
    train(config.TRAINING_LOGS_DIR, model, manager, n_epochs=50)


def macro_example():
    # macro example
    manager = MathQAManager(root_dir=config.MATHQA_DIR, max_vocabulary_size=1000, dummy=True,
                            macro_file=config.MACRO_10_FILE)
    model = Seq2Seq(
        source_vocabulary_size=manager.text_vocabulary_size,
        target_vocabulary_size=manager.code_vocabulary_size,
        hidden_dim=32,
        pad_index=manager.pad_index
    )
    train(config.TRAINING_LOGS_DIR, model, manager, n_epochs=50)


def no_punctuation_example():
    manager = MathQAManager(root_dir=config.MATHQA_DIR, max_vocabulary_size=1000, dummy=True,
                            no_punctuation=True, macro_file=config.MACRO_10_FILE)
    model = Seq2Seq(
        source_vocabulary_size=manager.text_vocabulary_size,
        target_vocabulary_size=manager.code_vocabulary_size,
        hidden_dim=32,
        pad_index=manager.pad_index
    )
    train(config.TRAINING_LOGS_DIR, model, manager, n_epochs=500, evaluate_every=20)


if __name__ == "__main__":
    # simple_example()
    # macro_example()
    no_punctuation_example()
