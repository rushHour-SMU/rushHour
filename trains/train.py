import random
import torch
import torch.optim as optim
import torch.nn as nn
from agents.dqn_agent import DQN
from agents.replay_buffer import ReplayBuffer


def train_dqn(env, num_episodes=100, batch_size=128, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=150):
    # 모델 초기화
    obs_shape = env.observation_space.shape[0] * env.observation_space.shape[1]
    n_actions = env.action_space.nvec[0] * env.action_space.nvec[1]
    policy_net = DQN(obs_shape, n_actions)
    target_net = DQN(obs_shape, n_actions)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-2)
    replay_buffer = ReplayBuffer(max_size=10000)

    epsilon = epsilon_start
    rewards_history = []
    steps_history = []
    last_episode_steps = []  # 마지막 에피소드의 행동 저장

    max_reward = float('-inf')  # 가장 높은 보상 초기화
    max_reward_episode = -1    # 가장 높은 보상이 발생한 에피소드

    for episode in range(num_episodes):
        state = env.reset()
        state = state.flatten()
        total_reward = 0
        steps = 0
        done = False

        episode_steps = []  # 현재 에피소드의 행동 저장

        while not done:
            # Epsilon-Greedy 행동 선택
            if random.random() < epsilon:
                action = [random.randint(0, env.action_space.nvec[0] - 1),
                          random.randint(0, env.action_space.nvec[1] - 1)]
            else:
                with torch.no_grad():
                    q_values = policy_net(torch.tensor(state, dtype=torch.float).unsqueeze(0))
                    action = torch.argmax(q_values).item()
                    action = [action // env.action_space.nvec[1], action % env.action_space.nvec[1]]

            # 환경에서 한 스텝 실행
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.flatten()

            # 현재 step 데이터 저장
            if episode == num_episodes - 1:
                episode_steps.append((state, action))  # 상태와 행동 저장

            # 리플레이 메모리에 저장
            replay_buffer.push(state, action[0] * env.action_space.nvec[1] + action[1], reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

            # 학습
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                # Q-Learning 대상 계산
                q_values = policy_net(states)
                q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                next_q_values = target_net(next_states).max(1)[0]
                targets = rewards + gamma * next_q_values * (1 - dones)

                # 손실 계산 및 역전파
                loss = nn.functional.mse_loss(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 타겟 네트워크 업데이트
        if episode % 5 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Epsilon 감소
        epsilon = max(epsilon_end, epsilon - (epsilon_start - epsilon_end) / epsilon_decay)

        rewards_history.append(total_reward)
        steps_history.append(steps)

        # 가장 높은 보상 갱신
        if total_reward > max_reward:
            max_reward = total_reward
            max_reward_episode = episode

        # 마지막 에피소드 step 데이터 저장
        if episode == num_episodes - 1:
            last_episode_steps = episode_steps

        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

    return rewards_history, steps_history, last_episode_steps, max_reward, max_reward_episode
