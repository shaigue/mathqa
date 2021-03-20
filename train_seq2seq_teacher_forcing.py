"""This is a code to train a sequence to sequence model with teacher forcing"""
from collections import defaultdict
import json

import torch
import torch.nn.functional as F
from mathqa_processing import MathQAManager, MathQADatapoint
from teacher_forcing_gru_encoder_decoder import Seq2Seq


def teacher_forcing_loss(target_token_indices, predicted_target_token_logits):
    seq_len, batch_size, target_vocabulary_size = predicted_target_token_logits.shape
    assert target_token_indices.shape == (seq_len, batch_size)
    # drop the first target symbol <SOS>
    target_token_indices = target_token_indices[1:]
    # drop the last predicted
    predicted_target_token_logits = predicted_target_token_logits[:-1]
    # swap the axis so that the probabilities over the vocabulary will be in dim=1
    predicted_target_token_logits = torch.transpose(predicted_target_token_logits, 2, 1)
    # output.shape = (N, C, ...), target.shape = (N, ...)
    return F.cross_entropy(predicted_target_token_logits, target_token_indices)


def tensorize_token_list(token_list: list[int], device):
    return torch.tensor(token_list, dtype=torch.int64, device=device).view(-1, 1)


def train_datapoint(model: Seq2Seq, optimizer: torch.optim.Optimizer, datapoint: MathQADatapoint, device):
    optimizer.zero_grad()

    source_token_indices = datapoint.text_token_indices
    source_token_indices = tensorize_token_list(source_token_indices, device)

    target_token_indices = datapoint.code_token_indices
    target_token_indices = tensorize_token_list(target_token_indices, device)

    predicted_target_token_logits = model(source_token_indices, target_token_indices)

    loss = teacher_forcing_loss(target_token_indices, predicted_target_token_logits)

    loss.backward()
    optimizer.step()

    return loss.detach().cpu().item()


def train_epoch(model: Seq2Seq, optimizer: torch.optim.Optimizer, mathqa_manager: MathQAManager, device):
    model.train()
    total_loss = 0
    n_batches = 0

    for datapoint_index, datapoint in enumerate(mathqa_manager.get_dataset_iterator('train', shuffle=True)):
        batch_loss = train_datapoint(model, optimizer, datapoint, device)
        total_loss += batch_loss
        n_batches += 1

    avg_loss = total_loss / n_batches
    return avg_loss


def evaluate_datapoint(model: Seq2Seq, manager: MathQAManager, datapoint: MathQADatapoint, device) -> bool:
    source_token_indices = datapoint.text_token_indices
    source_token_indices = tensorize_token_list(source_token_indices, device)
    generated_token_indices = model.generate(
        source_token_indices,
        start_of_string_token_index=manager.code_start_token_index,
        end_of_string_token_index=manager.code_end_token_index,
        max_target_seq_len=manager.code_max_len,
    )
    return manager.check_generated_code_correctness(generated_token_indices, datapoint)


def evaluate(model: Seq2Seq, manager: MathQAManager, partition: str, device) -> float:
    model.eval()
    total_correct = 0
    n_batches = 0

    with torch.no_grad():
        for datapoint_index, datapoint in enumerate(manager.get_dataset_iterator(partition, shuffle=False)):
            correct = evaluate_datapoint(model, manager, datapoint, device)
            if correct:
                total_correct += 1
            n_batches += 1

    correctness_rate = total_correct / n_batches
    return correctness_rate


def train(model: Seq2Seq, mathqa_manager: MathQAManager, n_epochs: int = 10, evaluate_every: int = 5,
          logs_file='training_logs.json', checkpoint_file='checkpoint.pt'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, verbose=True)

    best_dev_correctness_rate = 0
    logs = defaultdict(list)
    for epoch_index in range(n_epochs):
        epoch_loss = train_epoch(model, optimizer, mathqa_manager, device)
        print(f"epoch={epoch_index}, loss={epoch_loss}")

        logs['epoch_loss'].append({'epoch': epoch_index, 'value': epoch_loss})

        # checkpoint - save only the best performing model on dev
        if (epoch_index + 1) % evaluate_every == 0 or epoch_index == n_epochs - 1:
            print("evaluating...")
            train_correctness_rate = evaluate(model, mathqa_manager, 'train', device)
            print(f"train_correctness_rate={train_correctness_rate}")
            dev_correctness_rate = evaluate(model, mathqa_manager, 'dev', device)
            print(f"dev_correctness_rate={dev_correctness_rate}")
            print("finished evaluating.")

            logs['train_correctness_rate'].append({'epoch': epoch_index, 'value': train_correctness_rate})
            logs['dev_correctness_rate'].append({'epoch': epoch_index, 'value': dev_correctness_rate})
            # check if to save the model
            if dev_correctness_rate > best_dev_correctness_rate:
                print("saving best model checkpoint...")
                torch.save(model.state_dict(), checkpoint_file)
                best_dev_correctness_rate = dev_correctness_rate
            # update the scheduler
            lr_scheduler.step(dev_correctness_rate)

    print("training done.")
    print("evaluating on test...")
    # make sure to load the best performing model on dev
    model.load_state_dict(torch.load(checkpoint_file))
    test_correctness_rate = evaluate(model, mathqa_manager, 'test', device)
    print(f"test_correctness_rate={test_correctness_rate}")
    logs['test_correctness_rate'].append(test_correctness_rate)
    # save logs
    with open(logs_file, 'w') as f:
        json.dump(logs, f)
    print("evaluation on test done.")
    print("finished.")


def example():
    import config
    manager = MathQAManager(root_dir=config.MATHQA_DIR, max_vocabulary_size=1000, dummy=True)
    model = Seq2Seq(
        source_vocabulary_size=manager.text_vocabulary_size,
        target_vocabulary_size=manager.code_vocabulary_size,
        hidden_dim=32
    )
    train(model, manager, n_epochs=300)


if __name__ == "__main__":
    example()
