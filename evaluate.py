# 학습시킨 에이전트의 성능을 테스트하기 위한 코드

import gym
import yaml
import torch
from agents.dqn_agent import DQNAgent

# Config 파일 로드
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# 환경과 에이전트 초기화
env = gym.make(config["env_name"])
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQNAgent(state_dim, action_dim, config)

# 학습된 모델 로드
agent.model.load_state_dict(torch.load("saved_model.pth"))  # 모델 파일 경로는 저장된 모델 경로로 대체

# 평가 파라미터
num_episodes = 10  # 평가할 에피소드 수

# 평가 루프
total_rewards = []

for episode in range(num_episodes):
    state = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        state = next_state

    total_rewards.append(episode_reward)
    print(f"Episode {episode + 1}, Total Reward: {episode_reward}")

# 평균 보상 출력
avg_reward = sum(total_rewards) / num_episodes
print(f"Average Reward over {num_episodes} episodes: {avg_reward}")
