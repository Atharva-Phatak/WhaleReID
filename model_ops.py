import torch
from arch.model import EmbeddingNet
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from utils.schedulers import WarmupCosineLambda
from dagster import In, Out, op
from omegaconf import DictConfig
from typing import Any
from loss.loss import CosineMarginCrossEntropy, ArcFaceLossAdaptiveMargin
from metrics.precision import TopKPrecision
import numpy as np

@op(ins={"params": In(DictConfig), "counts": In(Any)})
def get_criterion(params, counts = None):
    if params.loss_name == "cosine_margin":
        return CosineMarginCrossEntropy(s=params.s, m=params.m)
    else:
        margins_id = np.power(counts, params.margin_power) * params.margin_coef + params.margin_cons_id
        return ArcFaceLossAdaptiveMargin(margins_id, params.num_classes, params.s)



@op
def get_metric():
    return TopKPrecision(top_k=(1, 5))


@op
def get_scaler():
    return torch.cuda.amp.GradScaler()


@op(ins={"model": In(torch.nn.Module), "params": In(DictConfig)})
def configure_optimizer(model, params):
    # train all layers
    other_parameters = [
        param
        for name, param in model.named_parameters()
        if ("norm_linear" not in name and "neck" not in name)
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": model.norm_linear.parameters(), "lr": params.last_linear_lr},
            {"params": model.neck.parameters(), "lr": params.last_linear_lr},
            {"params": other_parameters},
        ],
        lr=params.lr,
        weight_decay=5e-4,
    )
    return optimizer


@op(
    ins={"params": In(DictConfig)},
    out={
        "model": Out(torch.nn.Module),
        "optimizer": Out(torch.optim.Optimizer),
        "scheduler": Out(Any),
    },
)
def compile_model(params):
    """Method to compile model and setup optimizers/schedulers"""
    model = EmbeddingNet(params)
    optimizer = configure_optimizer(model, params)
    warmup_steps = params.epochs * params.warmup_steps_ratio
    cycle_steps = params.epochs - warmup_steps
    lr_lambda = WarmupCosineLambda(warmup_steps, cycle_steps, params.lr_decay_scale)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return model, optimizer, scheduler
