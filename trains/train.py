import random
import torch
import torch.optim as optim
import torch.nn as nn
from agents.dqn_agent import DQN
from agents.replay_buffer import ReplayBuffer
import numpy as np
def train_dqn(env, num_episodes=100, batch_size=128, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=150):
    obs_shape = env.observation_space.shape[0] * env.observation_space.shape[1]
    n_actions = env.action_space.nvec.prod().item()
    policy_net = DQN(obs_shape, n_actions)
    target_net = DQN(obs_shape, n_actions)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-2)
    replay_buffer = ReplayBuffer(max_size=10000)
    loss_fn = nn.MSELoss()

    epsilon = epsilon_start
    rewards_history = []
    steps_history = []
    last_episode_steps = []

    for episode in range(num_episodes):
        state = env.reset().flatten().astype(np.float32)
        state = torch.tensor(state)
        total_reward = 0
        steps = 0

        episode_steps = []

        while True:
            if random.random() < epsilon:
                action = random.randrange(n_actions)
            else:
                with torch.no_grad():
                    q_values = policy_net(state.unsqueeze(0))
                    action = q_values.argmax().item()

            next_state, reward, done, _ = env.step([action // env.action_space.nvec[1], action % env.action_space.nvec[1]])
            next_state = torch.tensor(next_state.flatten().astype(np.float32))

            episode_steps.append((state.numpy(), action, reward, next_state.numpy(), done))

            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                q_values = policy_net(states).gather(1, actions.unsqueeze(1))
                next_q_values = target_net(next_states).max(1)[0].detach()

                # 수정된 부분: `1 - dones`를 `1 - dones.float()`로 변경
                targets = rewards + (gamma * next_q_values * (1 - dones.float()))

                loss = loss_fn(q_values, targets.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        if episode % 5 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(epsilon_end, epsilon - (epsilon_start - epsilon_end) / epsilon_decay)
        rewards_history.append(total_reward)
        steps_history.append(steps)

        if episode == num_episodes - 1:
            last_episode_steps = episode_steps

        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

    return rewards_history, steps_history, last_episode_steps
