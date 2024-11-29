import random
import torch
import torch.optim as optim
from agent import DQN
from replay_buffer import ReplayBuffer


def train_dqn(env, num_episodes=100, batch_size=128, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=100):
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

    for episode in range(num_episodes):
        state = env.reset()
        state = state.flatten()
        total_reward = 0
        done = False

        while not done:
            if random.random() < epsilon:
                action = [random.randint(0, env.action_space.nvec[0] - 1),
                          random.randint(0, env.action_space.nvec[1] - 1)]
            else:
                with torch.no_grad():
                    q_values = policy_net(torch.tensor(state, dtype=torch.float).unsqueeze(0))
                    action = torch.argmax(q_values).item()
                    action = [action // env.action_space.nvec[1], action % env.action_space.nvec[1]]

            next_state, reward, done, _ = env.step(action)
            next_state = next_state.flatten()

            replay_buffer.push(state, action[0] * env.action_space.nvec[1] + action[1], reward, next_state, done)

            state = next_state
            total_reward += reward

            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                q_values = policy_net(states)
                q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                next_q_values = target_net(next_states).max(1)[0]
                targets = rewards + gamma * next_q_values * (1 - dones)

                loss = nn.functional.mse_loss(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if episode % 5 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(epsilon_end, epsilon - (epsilon_start - epsilon_end) / epsilon_decay)

        rewards_history.append(total_reward)
        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

    return rewards_history
