import torch
import torch.nn as nn


class Module(nn.Module):
    def __init__(
        self, 
        n_users: int, 
        n_items: int, 
        n_factors: int, 
        alpha: float,
        trn_pos_per_user: torch.Tensor,
    ):
        super(Module, self).__init__()
        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # device setting
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(DEVICE)

        # global attr
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.alpha = alpha
        self.trn_pos_per_user = trn_pos_per_user.to(self.device)

        # generate layers
        self._init_layers()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        user_idx: (B,)
        item_idx: (B,)
        """       
        return self._score(user_idx, item_idx)

    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        user_idx: (B,)
        item_idx: (B,)
        """  
        with torch.no_grad():
            logit = self._score(user_idx, item_idx)
            pred = torch.sigmoid(logit)
        return pred

    def _score(self, user_idx, item_idx):
        # Get embeddings
        context_slice = self._context(user_idx, item_idx)                # (B, H, D)
        target_slice = self.embed_target(item_idx)                       # (B, D)

        # Dot product
        dot = (context_slice * target_slice).sum(dim=1, keepdim=True)    # (B, 1)

        # Bias
        b_u = self.bias_user(user_idx)                          # (B, 1)
        b_i = self.bias_item(item_idx)                          # (B, 1)

        # Final score
        logit = (b_u + b_i + dot).squeeze(-1)                   # (B, 1)
        
        return logit

    def _context(self, user_idx, item_idx):
        # (B, H): item IDs user interacted with (with padding = n_items)
        user_histories = self.trn_pos_per_user[user_idx]            # long tensor
        
        # mask to current target item from history
        mask_target = user_histories == item_idx.unsqueeze(1)       # (B, H)
        # mask to padding
        mask_padding = user_histories == self.n_items               # (B, H)
        # final mask
        mask = mask_target | mask_padding                           # (B, H)

        # Get embeddings
        hist_slice = self.embed_hist(user_histories)                # (B, H, D)

        # Apply mask
        mask = mask.unsqueeze(-1)                                   # (B, H, 1)
        masked_hist_slice = hist_slice * (~mask)                    # (B, H, D)

        # Sum and normalize
        sum_hist = masked_hist_slice.sum(dim=1)                 # (B, D)
        num_hist = mask.sum(dim=1).clamp(min=1e-8)              # (B, 1)
        context = sum_hist / num_hist.pow(self.alpha)           # (B, D)

        return context

    def _init_layers(self):
        # Item embeddings
        kwargs = dict(
            num_embeddings=self.n_items+1, 
            embedding_dim=self.n_factors,
            padding_idx=self.n_items,
        )
        self.embed_hist = nn.Embedding(**kwargs)            # p_j
        self.embed_target = nn.Embedding(**kwargs)          # q_i

        # Bias terms
        kwargs = dict(
            num_embeddings=self.n_users+1, 
            embedding_dim=1,
            padding_idx=self.n_users,
        )
        self.bias_user = nn.Embedding(**kwargs)
        
        kwargs = dict(
            num_embeddings=self.n_items+1, 
            embedding_dim=1,
            padding_idx=self.n_items,
        )
        self.bias_item = nn.Embedding(**kwargs)

        # # Init
        # nn.init.normal_(self.embed_hist.weight, std=0.01)
        # nn.init.normal_(self.embed_target.weight, std=0.01)
        # nn.init.zeros_(self.bias_user.weight)
        # nn.init.zeros_(self.bias_item.weight)