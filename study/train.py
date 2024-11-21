import gym
import numpy as np
from agents.dqn.agent import DQNAgent

# Toy Text 환경 초기화 (FrozenLake 사용)
env = gym.make("FrozenLake-v1", is_slippery=True)
state_dim = env.observation_space.n
action_dim = env.action_space.n

# 에이전트 초기화
agent = DQNAgent(state_dim, action_dim)

num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        # 행동 선택 및 환경 상호작용
        action = agent.select_action([state])
        next_state, reward, done, _ = env.step(action)
        
        # 학습 단계 수행
        agent.train_step([state], action, reward, [next_state], done)
        
        state = next_state
        total_reward += reward

    # epsilon 감소 및 타겟 네트워크 업데이트
    agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)
    if episode % 10 == 0:
        agent.update_target_network()
    
    # 학습 상황 출력
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

env.close()
