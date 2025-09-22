# utils.py
import torch
from config import EMBED_DIM, STATE_DIM, MAX_HISTORY_LEN
import numpy as np

def build_state_vector(user_idx, history_item_idxs, item_emb_module):
    """
    把 (user, 最近 history) 映射成state embedding：
    简单实现：user history embedding = average of item embeddings in history
    user_id embedding 暂时用 zero-vector（或可用单独 user embedding）：user数量庞大，用户画像转成embedding最合适
    最终 state = concat(user_emb, hist_emb) -> (STATE_DIM,)
    """
    device = next(item_emb_module.parameters()).device
    if len(history_item_idxs) == 0:
        hist_emb = torch.zeros(EMBED_DIM, device=device)
    else:
        idxs = torch.tensor(history_item_idxs, device=device, dtype=torch.long)
        hist_emb = item_emb_module(idxs).mean(dim=0)
    # simple user embedding placeholder (zeros) —— 可以加 user embedding 参数化
    user_emb = torch.zeros(EMBED_DIM, device=device)
    state = torch.cat([user_emb, hist_emb], dim=0)
    return state.detach()
