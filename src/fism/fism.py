import torch
from components.histories import Histories
from components.base import BaseModel
from .layers.embedding import IDXEmbeddingWithHistory
from .layers.pooling import MeanPoolingLayer
from .layers.matching import build as build_matching_layer
from .layers.prediction import ProjectionLayer


class FactoredItemSimilarityModels(BaseModel):
    def __init__(
        self, 
        histories: Histories,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        alpha: float,
    ):
        """
        Fism: factored item similarity models for top-n recommender systems (Kabbur et al., 2013)
        -----
        Implements the base structure of Factored Item Similarity Models (FSIM),
        MF & id embedding based collaborative filtering model.

        Args:
            num_users (int):
                total number of users in the dataset, U.
            num_items (int):
                total number of items in the dataset, I.
            embedding_dim (int):
                dimensionality of user and item latent representation vectors.
            alpha (float):
                history length normalization factor.
            histories (Histories): 
                historical item interactions for each user, represented as item indices.
                (shape: [U, history_length])
            """
        super().__init__(locals())

        # HISTORY IDX VIEWER ==========
        self.histories = histories

        # IDX EMBEDDING ==========
        self.embedding = IDXEmbeddingWithHistory(
            num_items=num_items,
            embedding_dim=embedding_dim,
        )

        # HISTORY POOLING ==========
        self.pooling = MeanPoolingLayer(
            alpha=alpha,
        )

        # BILINEAR MATCHING FUNCTION ==========
        self.matching = build_matching_layer(
            name="mf",
        )

        # PREDICTION ==========
        self.prediction = ProjectionLayer(
            dim=embedding_dim,
        )

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        # SEARCH HISTORY IDX ==========
        hist_idx, mask = self.histories(user_idx, item_idx)
        # IDX EMBEDDING ==========
        item_emb, hist_emb = self.embedding(item_idx, hist_idx)
        # HISTORY POOLING ==========
        user_pooled = self.pooling(hist_emb, mask)
        # BILINEAR MATCHING FUNCTION ==========
        X_pred = self.matching(user_pooled, item_emb)
        # PRED VEC ==========
        return X_pred

    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate Method
        -----

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])
        
        Returns:
            logit (torch.Tensor): (u,i) pair interaction logit (shape: [B,])
        """
        # INTERACTION MODELING
        X_pred = self.forward(user_idx, item_idx)
        # PREDICTION
        logit = self.prediction(X_pred)
        return logit
