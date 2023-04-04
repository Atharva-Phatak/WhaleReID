
import torch
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
from metrics.base import BaseTopKMetric

class TopKPrecision(BaseTopKMetric):
    def __init__(self, top_k = (1,)):
        super().__init__(top_k = top_k, metric_name = "precision")

    def __call__(self, output: torch.Tensor, y: torch.Tensor):
        res = {}
        with torch.no_grad():
            for k in self.top_k:
                score_array = torch.tensor([1.0 / i for i in range(1, k + 1)], device=output.device)
                topk = output.topk(k)[1]
                match_mat = topk == y[:, None].expand(topk.shape)
                precision = (match_mat * score_array).sum(dim=1)
                #print(precision)
                res[f"{self.metric_name}_{k}"] = precision.mean().detach().cpu().item()
        return res