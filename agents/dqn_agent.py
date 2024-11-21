import torch
import random
import numpy as np
from collections import deque
from models.q_network import QNetwork

class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # 신경망
        self.policy_net = QNetwork(state_size, action_size)
        self.target_net = QNetwork(state_size, action_size)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # 경험 리플레이 버퍼
        self.memory = deque(maxlen=2000)

        # Target 네트워크 초기화
        self.update_target_network()

    def update_target_network(self):
        """Target 네트워크 갱신"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def choose_action(self, state):
        """Epsilon-Greedy 행동 선택"""
        if np.random.rand() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        """경험 리플레이 버퍼에 저장"""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        """Replay Buffer에서 샘플링하여 학습"""
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)

            target = self.policy_net(state).clone().detach()
            if done:
                target[0][action] = reward
            else:
                with torch.no_grad():
                    t_next = self.target_net(next_state)
                    target[0][action] = reward + self.gamma * torch.max(t_next)

            output = self.policy_net(state)
            loss = torch.nn.MSELoss()(output, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
