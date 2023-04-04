from trainer.train import trainer_amp
from trainer.validation import validation_amp
from trainer.batch import minibatch_whale_embedder
from utils.utils import ModelCheckpointer, seed_everything, get_logger, create_parser, asMinutes, stdout_results, stdout_info, set_mlflow_vars
from data_ops import get_dataloaders, get_params
from model_ops import compile_model, get_scaler, get_criterion, get_metric
from dagster import op, In, Out, graph, config_mapping, job
from typing import List, Dict, Callable, Any
from omegaconf import DictConfig, OmegaConf
from uuid import uuid4
import torch
import mlflow
import time


@op(ins={"params": In(DictConfig)})
def set_env_vars(params):
    """Method to set proper environment variables and setup experiment"""
    seed_everything(params.seed)






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
def trainer(context, model, optimizer, scaler, scheduler, params, dataloaders):
    """Method to train the model"""
    #set variables
    set_mlflow_vars()
    run_id = uuid4()
    ##create logger
    logger = get_logger(filename=f"{params.logs_path}/{params.backbone}-{params.loss_name}-{run_id}")
    # Create training and validation functions
    train_metric = get_metric()
    val_metric = get_metric()
    #create loss function
    loss_fn = get_criterion(params, dataloaders["train"].dataset.counts)
    #create train and val function
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

    # Checkpointer and EarlyStopping
    checkpointer = ModelCheckpointer(mode="max", path=f"{params.save_dir}/{params.backbone}-{run_id}.pt")

    #Stdout information
    stdout_info(logger, params)
    # Set experiment
    mlflow.set_experiment(experiment_name=f"{params.backbone}-{params.loss_name}")
    #Start model training
    with mlflow.start_run(run_name=f"{params.backbone}-{params.loss_name}-{run_id}"):
        mlflow.log_params(OmegaConf.to_container(params))
        for epoch in range(params.epochs):
            start = time.time()
            train_metrics = minibatch_whale_embedder(
                step_fn=train_fn, dataloader=dataloaders["train"], stage="train", metric_name=train_metric.metric_name
            )
            val_metrics = minibatch_whale_embedder(
                step_fn=val_fn, dataloader=dataloaders["validation"], stage="val", metric_name=val_metric.metric_name
            )
            mlflow.log_metrics(train_metrics, step=epoch)
            mlflow.log_metrics(val_metrics, step=epoch)
            end_time = (time.time() - start) 
            stdout_results(
                logger=logger,
                metricList=[train_metrics, val_metrics],
                current_epoch=epoch,
                time=end_time,
                epochs = params.epochs
            )
            checkpointer.step(
                score=val_metrics[f"val_{val_metric.metric_name}_1"],
                model_state={
                    "model": model,
                    "epoch": epoch,
                    **val_metrics,
                    **train_metrics,
                },
             )
            mlflow.pytorch.log_model(model, f"{params.backbone}-{run_id}")


@job(config=simplified_config)
def runner():
    params = get_params()
    # Set env variables
    set_env_vars(params)
    # Create dataloader
    dataloaders = get_dataloaders(params)
    # Create models / other stuff
    model, optimizer, scheduler = compile_model(params)
    scaler = get_scaler()
    #criterion = get_criterion(params, dataloaders["train"].dataset.counts)
    trainer(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        params=params,
        dataloaders=dataloaders,
    )


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    runner.execute_in_process(
        run_config={
            "path": args.path,
        }
    )
