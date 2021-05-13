"""train the graph generation model for mathqa"""
from dataclasses import dataclass
from typing import Union
import uuid

import torch
from ignite.handlers import global_step_from_engine
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ignite.engine import Events, Engine
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Average
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger

import config
from models.graph_generator import GraphGenerator, graph_generator_loss_fn, GraphGeneratorTrainOutput
from data_utils.mathqa_graph_generation_data import get_dataloader, GraphGenerationDatapoint
from program_processing.common import get_n_node_labels, get_max_n_args, get_node_label_to_edge_types, \
    get_op_label_indices, get_partial_graph_tensor


@dataclass
class TrainStepOutput:
    loss: float
    output: GraphGeneratorTrainOutput
    dp: GraphGenerationDatapoint


@dataclass
class ValStepOutput:
    loss: float
    output: GraphGeneratorTrainOutput
    dp: GraphGenerationDatapoint
    generated_adj_tensor: Tensor
    generated_node_labels: Tensor


def edge_accuracy_transform(step_out: Union[TrainStepOutput, ValStepOutput]) -> tuple[Tensor, Tensor]:
    return step_out.output.edge_logits, step_out.dp.edge_targets


def node_accuracy_transform(step_out: Union[TrainStepOutput, ValStepOutput]) -> tuple[Tensor, Tensor]:
    return step_out.output.node_logits, step_out.dp.node_targets


def is_exact_match(generated_adj_tensor: Tensor, generated_node_labels: Tensor,
                   true_adj_tensor: Tensor, true_node_labels: Tensor) -> bool:
    if generated_adj_tensor.shape != true_adj_tensor.shape:
        return False
    if generated_node_labels.shape != true_node_labels.shape:
        return False
    adj_match = torch.allclose(generated_adj_tensor, true_adj_tensor)
    node_match = torch.allclose(generated_node_labels, true_node_labels)
    return node_match and adj_match


def exact_match_from_val_step(step_out: ValStepOutput) -> float:
    return float(is_exact_match(
        generated_adj_tensor=step_out.generated_adj_tensor,
        generated_node_labels=step_out.generated_node_labels,
        true_adj_tensor=step_out.dp.adj_tensor,
        true_node_labels=step_out.dp.node_labels
    ))


class Accuracy:
    def __init__(self, output_transform):
        self.output_transform = output_transform

    def __call__(self, output):
        y_pred, y = self.output_transform(output)
        y_pred = torch.argmax(y_pred, dim=-1)
        correct = torch.eq(y_pred, y)
        return torch.sum(correct) / y.shape[0]


def train():
    # TODO: enable loading model from file
    # dummy = True
    dummy = False
    exp_name = 'graph_generator'
    exp_dir = config.get_exp_dir_path(exp_name)
    unique_id = uuid.uuid1()
    exp_dir = exp_dir / str(unique_id)
    logs_dir = exp_dir / 'runs'
    print(f"tensorboard --logdir=\"{logs_dir}\"", flush=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    train_loader = get_dataloader('train', dummy=dummy)
    val_loader = get_dataloader('dev', dummy=dummy)

    eval_every = 5

    n_node_labels = get_n_node_labels()
    n_edge_types = get_max_n_args()
    node_embedding_dim = 128
    condition_dim = config.TEXT_VECTOR_DIM
    node_label_to_edge_types = get_node_label_to_edge_types()
    node_labels_to_generate = get_op_label_indices()
    n_prop_steps = 3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphGenerator(
        n_node_labels=n_node_labels,
        n_edge_types=n_edge_types,
        node_embedding_dim=node_embedding_dim,
        condition_dim=condition_dim,
        node_label_to_edge_types=node_label_to_edge_types,
        node_labels_to_generate=node_labels_to_generate,
    )
    model = model.to(device)
    optimizer = Adam(model.parameters())
    lr_scheduler = ReduceLROnPlateau(optimizer)

    def train_step(engine, dp: GraphGenerationDatapoint) -> TrainStepOutput:
        model.train()
        optimizer.zero_grad()
        dp = dp.to(device)

        output = model(
            adj_tensor=dp.adj_tensor,
            node_labels=dp.node_labels,
            start_node_i=dp.start_node_i,
            n_prop_steps=n_prop_steps,
            condition_vector=dp.condition_vector,
        )

        loss = graph_generator_loss_fn(output, dp.edge_targets, dp.node_targets)
        loss.backward()
        optimizer.step()

        return TrainStepOutput(loss.item(), output, dp)

    def validation_step(engine, dp: GraphGenerationDatapoint) -> ValStepOutput:
        model.eval()

        with torch.no_grad():
            dp = dp.to(device)
            output = model(
                adj_tensor=dp.adj_tensor,
                node_labels=dp.node_labels,
                start_node_i=dp.start_node_i,
                n_prop_steps=n_prop_steps,
                condition_vector=dp.condition_vector,
            )
            loss = graph_generator_loss_fn(output, dp.edge_targets, dp.node_targets)
            partial_adj_tensor, partial_node_labels = get_partial_graph_tensor(
                adj_tensor=dp.adj_tensor,
                node_labels=dp.node_labels,
                start_node_i=dp.start_node_i
            )
            generated_adj_tensor, generated_node_labels = model.generate(
                partial_adj_tensor=partial_adj_tensor,
                partial_node_labels=partial_node_labels,
                condition_vector=dp.condition_vector,
                n_prop_steps=n_prop_steps,
            )

        return ValStepOutput(
            loss=loss.item(),
            output=output,
            dp=dp,
            generated_adj_tensor=generated_adj_tensor,
            generated_node_labels=generated_node_labels,
        )

    trainer = Engine(train_step)
    train_evaluator = Engine(validation_step)
    dev_evaluator = Engine(validation_step)

    train_metric_dict = {
        'edge_accuracy': Average(output_transform=Accuracy(edge_accuracy_transform)),
        'node_accuracy': Average(output_transform=Accuracy(node_accuracy_transform)),
        'loss': Average(output_transform=lambda x: x.loss)
    }

    val_metric_dict = {
        'edge_accuracy': Average(output_transform=Accuracy(edge_accuracy_transform)),
        'node_accuracy': Average(output_transform=Accuracy(node_accuracy_transform)),
        'loss': Average(output_transform=lambda x: x.loss),
        'exact_match': Average(output_transform=exact_match_from_val_step)
    }

    for metric_name, metric in train_metric_dict.items():
        metric.attach(trainer, metric_name)

    for metric_name, metric in val_metric_dict.items():
        metric.attach(train_evaluator, metric_name)
        metric.attach(dev_evaluator, metric_name)

    train_pbar = ProgressBar(persist=True, desc='training')
    train_pbar.attach(trainer)
    train_eval_pbar = ProgressBar(persist=True, desc='train validation')
    train_eval_pbar.attach(train_evaluator)
    dev_eval_pbar = ProgressBar(persist=True, desc='dev validation')
    dev_eval_pbar.attach(dev_evaluator)

    @trainer.on(Events.EPOCH_COMPLETED(every=eval_every))
    def evaluate():
        train_evaluator.run(train_loader)
        dev_evaluator.run(val_loader)

    tb_logger = TensorboardLogger(log_dir=logs_dir)
    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        tag='training',
        metric_names=list(train_metric_dict.keys()),
        global_step_transform=global_step_from_engine(trainer)
    )
    tb_logger.attach_output_handler(
        train_evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag='train-validation',
        metric_names=list(val_metric_dict.keys()),
        global_step_transform=global_step_from_engine(trainer)
    )
    tb_logger.attach_output_handler(
        dev_evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag='dev-validation',
        metric_names=list(val_metric_dict.keys()),
        global_step_transform=global_step_from_engine(trainer)
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def lr_scheduler_step(engine: Engine):
        lr_scheduler.step(engine.state.metrics['loss'])

    def model_checkpoint_score_func(engine: Engine) -> float:
        return engine.state.metrics['exact_match']

    model_checkpoint = ModelCheckpoint(
        dirname=exp_dir,
        filename_prefix='best',
        score_function=model_checkpoint_score_func,
    )
    dev_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {'model': model})

    trainer.run(train_loader, max_epochs=400)


if __name__ == "__main__":
    train()
