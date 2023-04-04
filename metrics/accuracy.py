import torch
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
from metrics.base import BaseTopKMetric


class TopKAccuracy(BaseTopKMetric):
    def __init__(self, top_k = (1,)):
        super().__init__(top_k=top_k, metric_name="accuracy")

    def __call__(self, output, target):
        with torch.no_grad():
            maxk = max(self.top_k)
            batch_size = target.size(0)
            _, y_pred = output.topk(k=maxk, dim=1)
            y_pred = y_pred.t()
            target_reshaped = target.view(1, -1).expand_as(y_pred)
            correct = y_pred == target_reshaped
            # -- get topk accuracy
            #print(y_pred)
            #print(target_reshaped)
            res = {}  # idx is topk1, topk2, ... etc
            for k in self.top_k:
                # get tensor of which topk answer was right
                ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
                # flatten it to help compute if we got it correct for each example in batch
                flattened_indicator_which_topk_matched_truth = (
                    ind_which_topk_matched_truth.reshape(-1).float()
                )  # [k, B] -> [kB]
                # get if we got it right for any of our top k prediction for each example in batch
                tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(
                    dim=0, keepdim=True
                )  # [kB] -> [1]
                # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
                #print(tot_correct_topk)
                topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
                res[f"{self.metric_name}_{k}"] = topk_acc.item()
            return res  # list of topk accuracies for entire batch [topk1, topk2, ... etc]



