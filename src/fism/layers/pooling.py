import torch
import torch.nn as nn


class MeanPoolingLayer(nn.Module):
    def __init__(
        self,
        alpha: float,
    ):
        super().__init__()

        # global attr
        self.alpha = alpha

    def forward(
        self, 
        hist_emb: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # context vector: (B,H,D) -> (B,D)
        context = hist_emb.sum(dim=1)
        # calculate hist per anchor: (B,)
        hist_len = (~mask).sum(dim=1).clamp(min=1).pow(self.alpha)
        # apply normalized factor: (B,D)
        context /= (hist_len).unsqueeze(-1)
        return context