# 리플레이 버퍼: 에이전트가 환경과 상호작용한 경험을 저장하고, 이를 나중에 샘플링하여 학습에 사용합니다.
# 로깅: 학습 진행 중 로그를 기록하여 에피소드별 보상, 손실 값 등을 추적하고 기록하는 기능입니다.
# 기타 유틸리티 함수: 파일 저장/불러오기, 데이터 전처리, 그래프 그리기 등 다양한 보조 기능을 구현할 수 있습니다.
# 위와 같은 보조적인 함수 모듈 저장
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
장