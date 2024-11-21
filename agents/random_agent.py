import random

class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def choose_action(self):
        """랜덤하게 행동을 선택"""
        return tuple(random.randint(0, n - 1) for n in self.action_space.nvec)
