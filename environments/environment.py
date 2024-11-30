import gym
import numpy as np
import pygame
from gym import spaces
from collections import deque

class Vehicle:
    def __init__(self, x, y, length, orientation, movable=True):
        self.x = x
        self.y = y
        self.length = length
        self.orientation = orientation
        self.movable = movable

    def get_positions(self):
        positions = []
        if self.orientation == 'H':
            positions = [(self.x, self.y + i) for i in range(self.length)]
        elif self.orientation == 'V':
            positions = [(self.x + i, self.y) for i in range(self.length)]
        return positions
class RushHourEnv(gym.Env):
    def __init__(self, vehicle_data, grid_size=(8, 8), max_steps=1000):
        super(RushHourEnv, self).__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps = 0
        self.window_size = 400
        self.cell_size = self.window_size // self.grid_size[0]
        self.screen = None

        # 차량 초기화
        self.vehicle_data = vehicle_data
        self.vehicles = [
            Vehicle(data["x"], data["y"], data["length"], data["orientation"], data["movable"])
            for data in self.vehicle_data
        ]
        self.target_vehicle = self.vehicles[0]  # 첫 번째 차량은 타겟 차량
        self.exit_wall = (3, self.grid_size[1] - 1)  # 출구 위치
        self.walls = self.create_walls()

        # Gym 행동 및 관찰 공간 정의
        self.action_space = spaces.MultiDiscrete([len(self.vehicles), 2])  # (차량 번호, 방향)
        self.observation_space = spaces.Box(low=0, high=1, shape=self.grid_size, dtype=np.int32)

    def create_walls(self):
        """격자의 외곽 벽 생성"""
        walls = set()
        for i in range(self.grid_size[0]):
            walls.add((i, 0))  # 왼쪽 벽
            walls.add((i, self.grid_size[1] - 1))  # 오른쪽 벽
        for j in range(self.grid_size[1]):
            walls.add((0, j))  # 위쪽 벽
            walls.add((self.grid_size[0] - 1, j))  # 아래쪽 벽
        return walls

    def reset(self):
        self.steps = 0
        self.vehicles = [
            Vehicle(data["x"], data["y"], data["length"], data["orientation"], data["movable"])
            for data in self.vehicle_data
        ]
        self.target_vehicle = self.vehicles[0]
        return self.get_observation()

    def get_observation(self):
        """현재 상태를 반환"""
        grid = np.zeros(self.grid_size, dtype=np.int32)
        for i, vehicle in enumerate(self.vehicles, start=1):
            for pos in vehicle.get_positions():
                if 0 <= pos[0] < self.grid_size[0] and 0 <= pos[1] < self.grid_size[1]:
                    grid[pos] = i
        for wall in self.walls:
            grid[wall] = -1  # 벽
        return grid

    def step(self, action):
        vehicle_idx, direction = [int(a) for a in action]
        vehicle = self.vehicles[vehicle_idx]

        if vehicle.movable and self.move_vehicle(vehicle, direction):
            self.steps += 1
            reward = self.calculate_reward(vehicle)
            done = False

            # 타겟 차량이 출구에 도달했는지 확인
            if self.target_vehicle.get_positions()[-1] == (self.exit_wall[0], self.exit_wall[1] - 1):
                reward = 100
                done = True
            elif self.steps >= self.max_steps:
                done = True

            return self.get_observation(), reward, done, {}
        else:
            return self.get_observation(), -0.5, False, {}

    def move_vehicle(self, vehicle, direction):
        step = 1 if direction == 1 else -1
        if vehicle.orientation == 'H':
            new_positions = [(vehicle.x, vehicle.y + step + i) for i in range(vehicle.length)]
        elif vehicle.orientation == 'V':
            new_positions = [(vehicle.x + step + i, vehicle.y) for i in range(vehicle.length)]
        else:
            print(f"Invalid orientation for vehicle at ({vehicle.x}, {vehicle.y})")
            return False  # 잘못된 방향

        # 새 위치가 모두 유효한지 확인
        can_move = all(self.is_valid_position(vehicle, pos) for pos in new_positions)
        
        if can_move:
            if vehicle.orientation == 'H':
                vehicle.y += step
            elif vehicle.orientation == 'V':
                vehicle.x += step
            return True
        else:
            return False


    def is_valid_position(self, vehicle, position):
        """새로운 위치가 유효한지 확인"""
        x, y = position
        if position in self.walls:  # 벽과 충돌 확인
            return False
        if not (0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]):  # 격자 범위 확인
            return False
        for other_vehicle in self.vehicles:
            if other_vehicle != vehicle and position in other_vehicle.get_positions():  # 다른 차량과 충돌 확인
                return False
        return True

    def get_reachable_positions(self, start_positions):
        """BFS를 사용하여 타겟 차량이 이동 가능한 모든 위치를 탐색"""
        queue = deque(start_positions)
        visited = set(start_positions)
        reachable_positions = set()

        while queue:
            x, y = queue.popleft()
            reachable_positions.add((x, y))

            # 상하좌우 탐색
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_x, new_y = x + dx, y + dy
                new_pos = (new_x, new_y)

                if new_pos not in visited and self.is_valid_position(None, new_pos):
                    visited.add(new_pos)
                    queue.append(new_pos)

        return reachable_positions

    def calculate_reward(self, vehicle):
        """보상 계산"""
        target_positions = set(self.target_vehicle.get_positions())
        current_reachable = self.get_reachable_positions(target_positions)

        # 타겟 차량이 출구에 도달한 경우
        if self.target_vehicle.get_positions()[-1] == (self.exit_wall[0], self.exit_wall[1] - 1):
            return 100

        # 차량 이동 후 경로 차단 여부 확인
        if self.is_strategic_block(vehicle):
            return 10

        return -0.1  # 기본 페널티

    def is_strategic_block(self, vehicle):
        """차량 이동이 전략적 차단인지 확인"""
        target_positions = set(self.target_vehicle.get_positions())
        current_reachable = self.get_reachable_positions(target_positions)

        # 차량 이동 시뮬레이션
        self.simulate_vehicle_movement(vehicle, 1)
        new_reachable = self.get_reachable_positions(target_positions)

        # 차단으로 인해 이동 가능한 경로가 줄어든 경우
        if len(new_reachable) < len(current_reachable):
            return True
        return False

    def simulate_vehicle_movement(self, vehicle, steps=1):
        """차량의 이동을 시뮬레이션하여 상태를 임시 변경"""
        if vehicle.orientation == 'H':
            vehicle.y += steps
        elif vehicle.orientation == 'V':
            vehicle.x += steps

    def render(self, mode="human"):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Rush Hour Environment")

        self.screen.fill((255, 255, 255))  # 화면 초기화 (흰색 배경)

        for i, vehicle in enumerate(self.vehicles, start=1):
            color = (255, 0, 0) if vehicle == self.target_vehicle else (0, 0, 255)  # 타겟 차량은 빨간색
            x_start, y_start = vehicle.get_positions()[0]  # 차량의 시작 위치
            if vehicle.orientation == 'H':
                # 가로로 이어진 단일 블록으로 차량 그리기
                rect = pygame.Rect(
                    y_start * self.cell_size, x_start * self.cell_size,
                    self.cell_size * vehicle.length, self.cell_size
                )
            elif vehicle.orientation == 'V':
                # 세로로 이어진 단일 블록으로 차량 그리기
                rect = pygame.Rect(
                    y_start * self.cell_size, x_start * self.cell_size,
                    self.cell_size, self.cell_size * vehicle.length
                )
            pygame.draw.rect(self.screen, color, rect)  # 차량 색상
            pygame.draw.rect(self.screen, (0, 0, 0), rect, width=2)  # 테두리 그리기 (선명한 구분)

        for wall in self.walls:
            rect = pygame.Rect(wall[1] * self.cell_size, wall[0] * self.cell_size, self.cell_size, self.cell_size)
            color = (0, 255, 0) if wall == self.exit_wall else (0, 0, 0)  # 출구는 녹색
            pygame.draw.rect(self.screen, color, rect)

        pygame.display.update()