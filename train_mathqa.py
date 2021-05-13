"""This is a code to train a sequence to sequence model with teacher forcing"""
import json
from pathlib import Path
from typing import Union, Iterator

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from math_qa.math_qa import RawMathQAEntry
from preprocessing.mathqa_processing import MathQAManager, ErrorReport, ErrorType
from preprocessing.mathqa_torch_loader import get_loader, TrainBatch, EvalBatch
from models.simple_seq2seq import SimpleSeq2Seq

_logger = config.get_logger(__file__)


class TrainLogs:
    def __init__(self, partitions: list[str], metrics: list[str]):
        self.metrics = metrics
        self.partitions = partitions
        self.logs = {}
        # initialize empty dicts
        for part in partitions:
            self.logs[part] = {}
            for metric in metrics:
                self.logs[part][metric] = {}

    def update(self, partition: str, metric: str, epoch: int, value: float):
        assert partition in self.partitions
        assert metric in self.metrics
        assert epoch >= 0
        self.logs[partition][metric][epoch] = value

    def get(self, partition: str, metric: str, epoch: int) -> float:
        assert partition in self.partitions
        assert metric in self.metrics
        assert epoch >= 0
        return self.logs[partition][metric][epoch]

    def iter_epoch_value(self, partition: str, metric: str) -> Iterator[tuple[int, float]]:
        assert metric in self.metrics
        assert partition in self.partitions
        for epoch, value in self.logs[partition][metric].items():
            yield epoch, value

    def get_epoch_value_lists(self, partition: str, metric: str) -> tuple[list[int], list[float]]:
        epochs = []
        values = []
        for epoch, value in self.iter_epoch_value(partition, metric):
            epochs.append(epoch)
            values.append(value)
        return epochs, values

    def get_max_value(self, partition: str, metric: str) -> float:
        max_value = None
        for epoch, value in self.iter_epoch_value(partition, metric):
            if max_value is None or max_value < value:
                max_value = value
        return max_value

    def to_json(self, json_path: Path):
        with json_path.open('w') as f:
            json.dump(self.logs, f)

    @classmethod
    def from_json(cls, json_path: Path):
        with json_path.open('r') as f:
            logs: dict = json.load(f)
        # need to convert string epoch to int
        partitions = list(logs.keys())
        metrics = list(logs[partitions[0]].keys())
        new_logs = cls(partitions, metrics)
        for part, part_dict in logs.items():
            for metric, metric_dict in part_dict.items():
                for str_epoch, value in metric_dict.items():
                    int_epoch = int(str_epoch)
                    new_logs.update(part, metric, int_epoch, value)
        return new_logs


def get_train_log_path(train_logs_dir: Path) -> Path:
    return train_logs_dir / 'train_log.json'


def get_error_report_path(train_logs_dir: Path) -> Path:
    return train_logs_dir / 'error_report.json'


def get_checkpoint_path(train_logs_dir: Path) -> Path:
    return train_logs_dir / 'model.pt'


def teacher_forcing_loss(target_token_indices, predicted_target_token_logits, pad_index: int) -> torch.Tensor:
    seq_len, batch_size, target_vocabulary_size = predicted_target_token_logits.shape
    assert target_token_indices.shape == (seq_len, batch_size)
    # drop the first target symbol <SOS>
    target_token_indices = target_token_indices[1:]
    # drop the last predicted
    predicted_target_token_logits = predicted_target_token_logits[:-1]
    # swap the axis so that the probabilities over the vocabulary will be in key_dim=1
    predicted_target_token_logits = torch.transpose(predicted_target_token_logits, 2, 1)
    # output.shape = (N, C, ...), target.shape = (N, ...)
    return F.cross_entropy(predicted_target_token_logits, target_token_indices, ignore_index=pad_index)


def train_batch(model: SimpleSeq2Seq, optimizer: torch.optim.Optimizer, batch: TrainBatch) -> float:
    optimizer.zero_grad()

    predicted_target_token_logits = model(
        source_tokens=batch.text_tokens,
        target_tokens=batch.code_tokens,
        source_lens=batch.text_lens,
        target_lens=batch.code_lens
    )

    loss = teacher_forcing_loss(
        target_token_indices=batch.code_tokens,
        predicted_target_token_logits=predicted_target_token_logits,
        pad_index=model.pad_index
    )

    loss.backward()
    optimizer.step()

    return loss.item()


def train_epoch(model: SimpleSeq2Seq, optimizer: torch.optim.Optimizer, train_loader: DataLoader) -> float:
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in train_loader:
        batch_loss = train_batch(model, optimizer, batch)
        total_loss += batch_loss
        n_batches += 1

    avg_loss = total_loss / n_batches
    return avg_loss


def evaluate_batch(model: SimpleSeq2Seq, manager: MathQAManager, batch: EvalBatch,
                   beam_size: int = 1) -> list[ErrorReport]:

    generated = model.generate(
        source_tokens=batch.text_tokens,
        source_lens=batch.text_lens,
        start_of_sequence_token=manager.code_start_token_index,
        end_of_sequence_token=manager.code_end_token_index,
        max_target_seq_len=manager.code_max_len,
        beam_size=beam_size
    )
    error_reports = []
    for i in range(len(generated)):
        error_report = manager.get_error_report(
            generated=generated[i],
            inputs=batch.inputs[i],
            correct_answer=batch.answers[i],
        )
        error_reports.append(error_report)

    return error_reports


def evaluate(model: SimpleSeq2Seq, manager: MathQAManager, loader: DataLoader, return_per_sample=False,
             beam_size: int = 1) -> Union[float, tuple[float, list[ErrorReport]]]:
    model.eval()
    n_correct = 0
    n_samples = 0
    if return_per_sample:
        per_sample = []

    with torch.no_grad():
        for batch in loader:
            error_reports = evaluate_batch(model, manager, batch, beam_size)
            n_samples += len(error_reports)

            for er in error_reports:
                if er.error_type == ErrorType.no_error:
                    n_correct += 1

            if return_per_sample:
                per_sample += error_reports

    correctness_rate = n_correct / n_samples

    if return_per_sample:
        return correctness_rate, per_sample

    return correctness_rate


def convert_error_report_json(er: ErrorReport) -> dict:
    return {'error_type': er.error_type.name, 'generated_tokens': er.generated_tokens}


def train(dir_path: Path, model: SimpleSeq2Seq, manager: MathQAManager, n_epochs: int,
          evaluate_every=5, batch_size=32, lr=0.01, weight_decay=1e-4,
          lr_decay_factor=0.2, lr_patience=1, beam_size=1):

    # set up the logs files
    dir_path.mkdir(parents=True, exist_ok=True)
    logs_file = get_train_log_path(dir_path)
    error_reports_file = get_error_report_path(dir_path)
    checkpoint_file = get_checkpoint_path(dir_path)

    # decide on what device to use and move the model there
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # set up the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=lr_patience,
                                                              verbose=True, factor=lr_decay_factor)
    # prepare the data
    train_loader = get_loader(manager, device, 'train', train=True)
    eval_loaders = {
        part: get_loader(manager, device, part, train=False)
        for part in manager.partitions
    }

    # set up logging
    best_dev_correctness_rate = 0
    logs = TrainLogs(partitions=['train', 'dev', 'test'], metrics=['correctness_rate', 'loss'])
    _logger.info(f"using device={device}. starting training session {dir_path.name}.")

    for epoch_index in range(n_epochs):
        mean_loss = train_epoch(model, optimizer, train_loader)

        _logger.info(f"epoch={epoch_index}, mean_loss={mean_loss}")
        logs.update('train', 'loss', epoch_index, mean_loss)

        # checkpoint - save only the best performing model on dev
        if (epoch_index + 1) % evaluate_every == 0 or epoch_index == n_epochs - 1:
            for part in ['train', 'dev']:
                correctness_rate = evaluate(model, manager, eval_loaders[part], beam_size=beam_size)

                _logger.info(f"{part}_correctness_rate={correctness_rate}")
                logs.update(part, 'correctness_rate', epoch_index, correctness_rate)

            # check if to save the model
            dev_correctness_rate = logs.get('dev', 'correctness_rate', epoch_index)
            if dev_correctness_rate > best_dev_correctness_rate:
                _logger.info(f"saving best model checkpoint at epoch={epoch_index}")
                torch.save(model.state_dict(), checkpoint_file)
                best_dev_correctness_rate = dev_correctness_rate
            # update the scheduler
            lr_scheduler.step(dev_correctness_rate)

    # make sure to load the best performing model on dev
    model.load_state_dict(torch.load(checkpoint_file))
    # evaluate on each one of the partitions, and save their correctness rate and per_sample correctness in the logs
    partitions_error_reports = {}
    for part in manager.partitions:
        correctness_rate, error_reports = evaluate(
            model,
            manager,
            eval_loaders[part],
            return_per_sample=True,
            beam_size=beam_size
        )
        _logger.info(f"{part}_correctness_rate={correctness_rate}")
        logs.update(part, 'correctness_rate', n_epochs, correctness_rate)
        partitions_error_reports[part] = list(map(convert_error_report_json, error_reports))

    # save logs
    logs.to_json(logs_file)
    with error_reports_file.open('w') as f:
        json.dump(partitions_error_reports, f)

    _logger.info(f"{dir_path.name} training finished")


def get_manager(no_punctuation: bool = True, macro_file=None, dummy=False,
                raw_data: dict[str, list[RawMathQAEntry]] = None):
    return MathQAManager(
        root_dir=config.MATHQA_DIR,
        max_vocabulary_size=config.MAX_VOCABULARY_SIZE,
        dummy=dummy,
        macro_file=macro_file,
        no_punctuation=no_punctuation,
        raw_data=raw_data
    )


def get_model(manager: MathQAManager, dropout: float = 0, n_gru_layers: int = 1):
    return SimpleSeq2Seq(
        source_vocabulary_size=manager.text_vocabulary_size,
        target_vocabulary_size=manager.code_vocabulary_size,
        hidden_dim=config.INTERNAL_DIM,
        pad_index=manager.pad_index,
        dropout=dropout,
        n_gru_layers=n_gru_layers,
    )


def example():
    manager = get_manager(dummy=True)
    model = get_model(manager)
    train(config.TRAINING_LOGS_DIR, model, manager, 100)


def macro_example():
    manager = get_manager(macro_file=config.MACRO_10_FILE, dummy=True)
    model = get_model(manager)
    train(config.TRAINING_LOGS_DIR, model, manager, 100)


if __name__ == "__main__":
    example()
    # macro_example()
