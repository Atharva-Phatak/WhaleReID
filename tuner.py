import optuna
from dagster import In, Out, op
from metrics.precision import TopKPrecision
from trainer.train import trainer_amp
from trainer.validation import validation_amp
from trainer.batch import minibatch_whale_embedder
from omegaconf import DictConfig
import torch
from typing import Any


def objective(
    trial: optuna.trial.Trial,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.grad_scaler.GradScaler,
    scheduler: Any,
    loss_fn: Any,
    params: DictConfig,
    dataloaders: dict,
) -> float:
    params.m = trial.suggest_float("m", 0.35, 0.75)
    params.s = trial.suggest_float("s", 32.0, 50.0)
    if params.loss_name == "arc_margin":
        params.margin_power = trial.suggest_float("margin_power", -0.8, -0.05)
        params.margin_coef = trial.suggest_float("margin_coef", 0.2, 1.0)

    score = tuner_loop(
        trial=trial,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        loss_fn=loss_fn,
        params=params,
        dataloaders=dataloaders,
    )
    return score


def tuner_loop(
    trial, model, optimizer, scaler, scheduler, loss_fn, params, dataloaders
):
    # Create training and validation functions
    train_metric = TopKPrecision(top_k=(1, 5))
    val_metric = TopKPrecision(top_k=(1, 5))

    train_fn = trainer_amp(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        amp_scaler=scaler,
        loss_fn=loss_fn,
        metric=train_metric,
    )
    val_fn = validation_amp(
        model=model, amp_scaler=scaler, loss_fn=loss_fn, metric=val_metric
    )

    # Move model to gpu
    model = model.to(params.device)
    mean_precision_5 = 0
    for epoch in range(params.epochs):
        train_metrics = minibatch_whale_embedder(
            step_fn=train_fn, dataloader=dataloaders["train"], stage="train", metric_name=train_metric.metric_name
        )
        val_metrics = minibatch_whale_embedder(
            step_fn=val_fn, dataloader=dataloaders["validation"], stage="val", metric_name=val_metric.metric_name
        )
        mean_precision_5 += val_metrics["val_precision_5"]
        trial.report(val_metrics["val_precision_5"], epoch)

        if trial.should_prune():
            return optuna.exceptions.TrialPruned()

    return mean_precision_5 / params.epochs
