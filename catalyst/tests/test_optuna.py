# flake8: noqa

import os
from tempfile import TemporaryDirectory

import optuna
from pytest import mark
import torch
from torch import nn
from torch.utils.data import DataLoader

from catalyst import dl
from catalyst.contrib.datasets import MNIST
from catalyst.data.transforms import ToTensor
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES


def train_experiment(device):
    with TemporaryDirectory() as logdir:

        def objective(trial):
            lr = trial.suggest_loguniform("lr", 1e-3, 1e-1)
            num_hidden = int(trial.suggest_loguniform("num_hidden", 32, 128))

            loaders = {
                "train": DataLoader(
                    MNIST(os.getcwd(), train=True, download=True, transform=ToTensor()),
                    batch_size=32,
                ),
                "valid": DataLoader(
                    MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()),
                    batch_size=32,
                ),
            }
            model = nn.Sequential(
                nn.Flatten(), nn.Linear(784, num_hidden), nn.ReLU(), nn.Linear(num_hidden, 10)
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            runner = dl.SupervisedRunner(
                input_key="features", output_key="logits", target_key="targets"
            )
            runner.train(
                engine=dl.DeviceEngine(device),
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                loaders=loaders,
                callbacks={
                    "optuna": dl.OptunaPruningCallback(
                        loader_key="valid", metric_key="accuracy01", minimize=False, trial=trial
                    ),
                    "accuracy": dl.AccuracyCallback(
                        input_key="logits", target_key="targets", num_classes=10
                    ),
                },
                num_epochs=2,
            )
            score = runner.callbacks["optuna"].best_score
            return score

        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=1, n_warmup_steps=0, interval_steps=1
            ),
        )
        study.optimize(objective, n_trials=3, timeout=300)
        print(study.best_value, study.best_params)


def test_finetune_on_cpu():
    train_experiment("cpu")


@mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA device is not available")
def test_finetune_on_cuda():
    train_experiment("cuda:0")


@mark.skipif(
    not IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES < 2, reason="Number of CUDA devices is less than 2",
)
def test_finetune_on_cuda_device():
    train_experiment("cuda:1")
