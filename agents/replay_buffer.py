import random
from collections import deque
import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        # 데이터를 GPU 텐서로 받을 가능성이 있으므로 .cpu()를 호출해 변환
        state = state.cpu().numpy() if isinstance(state, torch.Tensor) else state
        next_state = next_state.cpu().numpy() if isinstance(next_state, torch.Tensor) else next_state
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, device=torch.device('cpu')):
        sample_items = random.sample(self.buffer, batch_size)
        states = torch.tensor([item[0] for item in sample_items], dtype=torch.float).to(device)
        actions = torch.tensor([item[1] for item in sample_items], dtype=torch.long).to(device)
        rewards = torch.tensor([item[2] for item in sample_items], dtype=torch.float).to(device)
        next_states = torch.tensor([item[3] for item in sample_items], dtype=torch.float).to(device)
        dones = torch.tensor([item[4] for item in sample_items], dtype=torch.bool).to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
