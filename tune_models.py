from optuna.integration.mlflow import MLflowCallback
from data_ops import get_dataloaders, get_params
from model_ops import compile_model, get_scaler, get_criterion, get_metric
from dagster import op, In, Out, graph, config_mapping, job
from utils.utils import _store_json, check_dirs, set_mlflow_vars, create_parser
from typing import List, Dict, Callable, Any
import optuna
from tuner import objective
import torch
from typing import Any
from omegaconf import DictConfig
import os
import json
import mlflow



@config_mapping(config_schema={"path": str})
def simplified_config(val):
    """Config mapping required by dagster job"""
    return {"ops": {"get_params": {"config": {"path": val["path"]}}}}


@op(
    ins={
        "model": In(torch.nn.Module),
        "optimizer": In(torch.optim.Optimizer),
        "scaler": In(torch.cuda.amp.grad_scaler.GradScaler),
        "scheduler": In(Any),
        "params": In(DictConfig),
        "dataloaders": In(Dict[str, torch.utils.data.DataLoader]),
    }
)
def optuna_tuner(model, optimizer, scaler, scheduler, params, dataloaders):
    set_mlflow_vars()
    # Create-mlflow-callback
    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(), metric_name="precision_5"
    )
    loss_fn = get_criterion(params, dataloaders["train"].dataset.counts)
    tuner = lambda trial: objective(
        trial=trial,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        loss_fn=loss_fn,
        params=params,
        dataloaders=dataloaders,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=params.seed),
        pruner=optuna.pruners.HyperbandPruner(),
        study_name=f"{params.backbone}-{params.loss_name}",
    )
    study.optimize(tuner, n_trials=10, callbacks=[mlflow_callback])
    trials = study.trials_dataframe()
    path = f"{params.trial_path}/{params.backbone}-{params.loss_name}-exp"
    check_dirs(path)
    trials.to_csv(f"{path}/{params.backbone}-trial.csv", index=False)
    _store_json(
        path=f"{path}/{params.backbone}-best_params.json", data=study.best_params
    )


@job(config=simplified_config)
def run_tuner():
    params = get_params()
    # Create dataloader
    dataloaders = get_dataloaders(params)
    # Create models / other stuff
    model, optimizer, scheduler = compile_model(params)
    scaler = get_scaler()
    optuna_tuner(model, optimizer, scaler, scheduler, params, dataloaders)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run_tuner.execute_in_process(
        run_config={
            "path": args.path
        }
    )
