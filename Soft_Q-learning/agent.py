# agent.py
"""
SoftQAgent：实现 Top-K candidate -> softmax over Q/alpha -> 采样 -> online TD 更新
关键函数：
- act(state_vec, candidate_item_idxs, candidate_item_embs)
- update(batch)
"""
import torch
import torch.nn.functional as F
from torch import optim
from models import QNetwork, ItemEmbedding
from replay import ReplayBuffer
from config import ALPHA, GAMMA, LR, BATCH_SIZE, POLYAK, UPDATES_PER_STEP, DEVICE

def soft_value_from_q(q, alpha):
    # q: (B, K)
    return alpha * torch.logsumexp(q / alpha, dim=1)  # (B,)

class SoftQAgent:
    def __init__(self, n_items, item_embedding, state_dim, alpha=ALPHA, gamma=GAMMA):
        self.alpha = alpha
        self.gamma = gamma
        self.item_emb = item_embedding  # ItemEmbedding instance (shared)
        self.qnet = QNetwork(state_dim, self.item_emb.emb.embedding_dim).to(DEVICE)
        self.qnet_target = QNetwork(state_dim, self.item_emb.emb.embedding_dim).to(DEVICE)
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        self.opt = optim.Adam(self.qnet.parameters(), lr=LR)
        self.replay = ReplayBuffer(capacity=200000)

    def act(self, state_vec, candidate_idxs):
        """
        state_vec: torch.tensor (state_dim,) or (B,state_dim)
        candidate_idxs: list/np.array of item idxs (K,)
        根据state和剩余的item，返回选中的 item idx, 以及 probs（用于日志）
        """
        device = DEVICE
        state = state_vec.unsqueeze(0).to(device) if state_vec.dim()==1 else state_vec.to(device)
        cidx = torch.tensor(candidate_idxs, device=device, dtype=torch.long).unsqueeze(0)  # (1,K)
        cemb = self.item_emb(cidx)  # (1,K,D)
        q_scores = self.qnet(state, cemb).squeeze(0)  # (K,)
        probs = F.softmax(q_scores / self.alpha, dim=0)  # (K,)
        # sample
        sel = torch.multinomial(probs, 1).item()
        return int(candidate_idxs[sel]), probs.detach().cpu().numpy(), q_scores.detach().cpu().numpy()

    def push_transition(self, s_vec, chosen_item_idx, reward, s_next_vec, next_cand_idxs):
        """
        store the embedding of chosen item & next candidate embs in replay buffer
        """
        chosen_emb = self.item_emb(torch.tensor(chosen_item_idx, device=DEVICE)).detach().cpu()
        next_cands = torch.tensor(next_cand_idxs, device=DEVICE, dtype=torch.long)
        next_cand_embs = self.item_emb(next_cands).detach().cpu()  # (K,D)
        self.replay.push((s_vec.cpu(), chosen_item_idx, chosen_emb.cpu(), float(reward), s_next_vec.cpu(), next_cand_embs.cpu()))

    def update(self, batch_size=BATCH_SIZE):
        """
        从replay buffer抽取batch size个sample更新policy model
        """
        if len(self.replay) < batch_size:
            return None
        batch = self.replay.sample(batch_size)
        # unpack
        s_bs = torch.stack([b[0] for b in batch]).to(DEVICE)         # (B, state_dim)
        chosen_item_idxs = torch.tensor([b[1] for b in batch], device=DEVICE, dtype=torch.long)  # (B,)
        # chosen_embs stored as cpu tensor
        chosen_embs = torch.stack([b[2] for b in batch]).to(DEVICE)  # (B, D)
        r_bs = torch.tensor([b[3] for b in batch], device=DEVICE)    # (B,)
        s_next = torch.stack([b[4] for b in batch]).to(DEVICE)      # (B, state_dim)
        next_cand_embs = torch.stack([b[5] for b in batch]).to(DEVICE)  # (B, K, D)

        # 当前状态 s 对所选动作 a 的 Q 预测
        q_pred = self.qnet(s_bs, chosen_embs)  # (B,)

        # 下一状态 s' 下所有候选动作的 Q 值
        q_next_all = self.qnet_target(s_next, next_cand_embs)  # (B, K)
        # 计算 V(s') = α logsumexp(Q/α)
        v_next = soft_value_from_q(q_next_all, self.alpha)     # (B,)
        # TD 目标
        y = r_bs + self.gamma * v_next
        # MSE(Qθ(s, a), y)
        loss = F.mse_loss(q_pred, y.detach())

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # polyak update target
        for p, pt in zip(self.qnet.parameters(), self.qnet_target.parameters()):
            pt.data.mul_(POLYAK)
            pt.data.add_((1 - POLYAK) * p.data)

        return loss.item()
