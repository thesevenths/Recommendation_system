# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import EMBED_DIM, STATE_DIM, DEVICE

class ItemEmbedding(nn.Module):
    def __init__(self, n_items, emb_dim=EMBED_DIM):
        super().__init__()
        self.emb = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.emb.weight, 0, 0.01)
    def forward(self, item_idx):
        return self.emb(item_idx)  # (B, D) or (D,)

class QNetwork(nn.Module):
    """
    Q(s, item) -> scalar
    state: we'll use concatenation of user_emb and history_emb (STATE_DIM)
    item_emb: EMBED_DIM
    """
    def __init__(self, state_dim=STATE_DIM, item_dim=EMBED_DIM, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + item_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
    def forward(self, state, item_emb):
        # state: (B, state_dim)
        # item_emb: (B, item_dim) or (B, K, item_dim)
        if item_emb.dim() == 3:
            B, K, D = item_emb.size()
            s_exp = state.unsqueeze(1).expand(-1, K, -1)  # (B,K,S)
            pair = torch.cat([s_exp, item_emb], dim=-1)  # (B,K,S+D)
            h = F.relu(self.fc1(pair))
            h = F.relu(self.fc2(h))
            return self.out(h).squeeze(-1)  # (B,K)
        else:
            pair = torch.cat([state, item_emb], dim=-1)
            h = F.relu(self.fc1(pair))
            h = F.relu(self.fc2(h))
            return self.out(h).squeeze(-1)  # (B,)
