import torch
import torch.nn as nn


class IDXEmbeddingWithHistory(nn.Module):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
    ):
        super().__init__()

        kwargs = dict(
            num_embeddings=num_items+2, 
            embedding_dim=embedding_dim,
            padding_idx=0,
        )
        self.item_emb = nn.Embedding(**kwargs)
        self.hist_emb = nn.Embedding(**kwargs)

        self.init_embeddings()

    def forward(
        self, 
        item_idx: torch.Tensor,
        hist_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        item_emb = self.item_emb(item_idx)
        hist_emb = self.hist_emb(hist_idx)
        return item_emb, hist_emb

    def init_embeddings(self):
        embeddings = [
            self.item_emb,
            self.hist_emb,
        ]

        for emb in embeddings:
            kwargs = dict(
                tensor=emb.weight, 
                mean=0.0, 
                std=0.01,
            )
            nn.init.normal_(**kwargs)