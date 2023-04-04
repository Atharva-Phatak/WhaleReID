import torch
from typing import Callable
def validation_amp(
    model: torch.nn.Module, amp_scaler, loss_fn: torch.nn.Module, metric: Callable
):
    def validate(batch):
        model.eval()
        inputs, targets, _ = batch
        inputs = inputs.to("cuda")
        targets = targets.to("cuda")
        with torch.cuda.amp.autocast():
            model_outputs = model(inputs)
            loss = loss_fn(model_outputs["cosine_logits"], targets)
        top_k = metric(model_outputs["cosine_logits"], targets)
        return loss, top_k

    return validate


def validation_base(model: torch.nn.Module, loss_fn: torch.nn.Module):
    def validate(batch):
        inputs, _, _ = batch
        model_outputs = model(inputs)
        loss = loss_fn(model_outputs, targets)
        return loss, model_outputs

    return validate
