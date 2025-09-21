# replay.py
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)
    def push(self, transition):
        # transition = (s_vec, item_idx, item_emb, r, s_next_vec, next_candidate_embs)
        self.buf.append(transition)
    def sample(self, batch_size):
        batch = random.sample(self.buf, min(batch_size, len(self.buf)))
        return batch
    def __len__(self):
        return len(self.buf)
