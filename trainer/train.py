import torch
from typing import Callable


def trainer_amp(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    amp_scaler,
    loss_fn: torch.nn.Module,
    metric: Callable,
    return_embedding: bool = True,
):
    def train(batch):
        model.train()
        inputs, targets, _ = batch
        inputs = inputs.to("cuda")
        targets = targets.to("cuda")
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            model_outputs = model(inputs, return_embedding=return_embedding)
            loss = loss_fn(model_outputs["cosine_logits"], targets)
        amp_scaler.scale(loss).backward()
        amp_scaler.step(optimizer)
        scheduler.step()
        amp_scaler.update()
        top_k = metric(model_outputs["cosine_logits"], targets)
        return loss, top_k

    return train


def trainer_base(
    model: torch.nn.Module,
    optimizer,
    scheduler,
    loss_fn: torch.nn.Module,
    metric: Callable,
):
    def train(batch):
        inputs, targets, _ = batch
        optimizer.zero_grad()
        model_outputs = model(inputs)
        loss = loss_fn(model_outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        return loss, model_outputs

    return train