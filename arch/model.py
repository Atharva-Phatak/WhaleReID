import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm
import math
from arch.layers import ArcMarginProductSubcenter, NormLinear




class EmbeddingNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        # self.params = params
        self.backbone, in_features = self._process_model(backbone=params.backbone)
        if params.embedding_size != -1:
            self.neck = self._create_neck(in_features, params)
        else:
            self.neck = nn.LayerNorm(in_features)
        
        if params.loss_name == "cosine_margin":
            self.norm_linear = NormLinear(in_features = in_features, out_features = params.num_classes)
        else:
            self.norm_linear = ArcMarginProductSubcenter(in_features = in_features, out_features = params.num_classes)

    def _create_neck(self, in_features, params):
        return nn.Sequential(
            *[
                nn.Linear(in_features, params.embedding_size),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(params.embedding_size),
            ]
        )

    def _process_model(self, backbone):
        model = timm.create_model(backbone, pretrained=True)
        last_layer = model.get_classifier()
        backbone_model = nn.Sequential(
            *[x for x in model.children() if x != last_layer]
        )
        #backbone_model.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        return backbone_model, last_layer.in_features

    def get_embedding(self, input):
        op = self.backbone(input)
        op = self.neck(op)
        return op

    def forward(self, input, return_embedding: bool = False):
        embedding = self.get_embedding(input)
        if return_embedding is True:
            return {
                "cosine_logits": self.norm_linear(embedding),
                "embeddings": embedding,
            }
        else:
            return {"cosine_logits": self.norm_linear(embedding)}
