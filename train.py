# 학습에 관련된 코드 예시

import gym
import yaml
from agents.dqn_agent import DQNAgent

# Load config
with open("config/config.yaml", "r") as file:

    config = yaml.safe_load(file)

env = gym.make(config["env_name"])
agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, config)

for episode in range(100):  # 예시로 100 에피소드 실행
    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
    print(f"Episode {episode} completed")
