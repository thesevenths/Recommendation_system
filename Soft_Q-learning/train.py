# train.py
"""
主训练脚本：
- 用 RecTmallLoader 读数据
- 初始化 item embedding / agent
- 流式遍历日志（以时间序），每条交互做：构造 state -> 用 agent 在 TopK candidates 上采样推荐 -> 计算 reward（若推荐 item == ground_truth，则为真实行为的 reward，否则 0）-> 存 transition -> 定期做 update
- 期间周期性输出在线累计指标（hit@1, hit@10, cumulative reward）
运行：
    python train.py
"""
import torch
import numpy as np
from config import *
from data_loader import RecTmallLoader
from models import ItemEmbedding
from agent import SoftQAgent
from utils import build_state_vector
from tqdm import tqdm

def action_to_reward(action):
    # 常见 mapping：若 action 是数值 1-4，按经验映射；若是字符串，按关键词映射
    try:
        # numeric
        an = float(action)
        # example mapping (可按需在 config 中调整)
        if an == 4:   # 假设 4 表示购买
            return 1.0
        elif an == 3: # 加购
            return 0.6
        elif an == 2: # 收藏
            return 0.3
        else:         # 浏览
            return 0.1
    except:
        s = str(action).lower()
        if "buy" in s or "purchase" in s or "pay" in s:
            return 1.0
        if "cart" in s:
            return 0.6
        if "fav" in s or "collect" in s or "like" in s:
            return 0.3
        return 0.1

def retrieve_topk_candidates(item_emb_module, state_vec, all_item_count, topk=TOPK):
    """
    简单暴力的 Top-K：计算 state_vector 与所有 item embedding 的点积相似度
    注意：数据量大时应换 FAISS
    """
    with torch.no_grad():
        # fetch all embeddings (may be large; for sample file it's fine)
        all_embs = item_emb_module.emb.weight.data  # (N, D)

        linear = torch.nn.Linear(128, 64)
        state_vec = linear(state_vec)  # 现在 state_vec 的维度是 (64,)
        # project state to item space via dot product with emb (这里直接用 dot)
        scores = torch.matmul(all_embs, state_vec.to(all_embs.device))  # (N,)
        topk_idx = torch.topk(scores, min(topk, scores.size(0))).indices.cpu().numpy().tolist()
        return topk_idx

def main():
    loader = RecTmallLoader(PRODUCT_CSV, LOG_CSV, REVIEW_CSV)
    loader.load()  # 加载全部 sample
    n_items = len(loader.item2idx)
    print(f"users={len(loader.user2idx)} items={n_items} logs={len(loader.logs)}")

    item_emb = ItemEmbedding(n_items).to(DEVICE)
    agent = SoftQAgent(n_items, item_emb, state_dim=STATE_DIM)

    # per-user recent history store (用于构造 state)
    user_hist = {uid:[] for uid in loader.user2idx.values()}

    cum_reward = 0.0
    hit1 = 0
    hit10 = 0
    total = 0

    # stream
    for step, (u_idx, item_idx, action, ts) in enumerate(tqdm(loader.stream_interactions(), total=len(loader.logs))):
        # build state vector: user embedding (we'll use average of past items) + history embedding
        hist = user_hist.get(u_idx, [])
        state_vec = build_state_vector(u_idx, hist, item_emb)  # torch.tensor (STATE_DIM,)

        # candidate generation: TopK + ensure ground truth in candidate set
        topk = retrieve_topk_candidates(item_emb, state_vec, n_items, topk=TOPK)
        if item_idx not in topk:
            topk[-1] = item_idx  # ensure ground truth included
        # agent act
        chosen_item, probs, q_scores = agent.act(state_vec, topk)

        # calculate reward
        true_reward = action_to_reward(action) if chosen_item == item_idx else 0.0

        # book-keeping
        cum_reward += true_reward
        total += 1
        if chosen_item == item_idx:
            hit1 += 1
        # for hit@10: if ground truth in topk (we ensured), count
        if item_idx in topk[:10]:
            hit10 += 1

        # next state: append the ground truth interaction into history
        if len(user_hist[u_idx]) >= MAX_HISTORY_LEN:
            user_hist[u_idx].pop(0)
        user_hist[u_idx].append(item_idx)
        s_next = build_state_vector(u_idx, user_hist[u_idx], item_emb)

        # push transition and update
        agent.push_transition(state_vec, chosen_item, true_reward, s_next, topk)
        if step > UPDATE_AFTER:
            for _ in range(UPDATES_PER_STEP):
                loss = agent.update()
        # periodic logging
        if step % 20000 == 0 and step>0:
            print(f"step={step} cum_reward={cum_reward:.2f} hit@1={hit1/total:.4f} hit@10={hit10/total:.4f}")

    print("finished")
    print(f"final cum_reward={cum_reward:.2f} hit@1={hit1/total:.4f} hit@10={hit10/total:.4f}")

if __name__ == "__main__":
    main()
