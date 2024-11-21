import gym
from gym import spaces
import numpy as np
import pygame


class Vehicle:
    """차량 클래스 정의"""
    def __init__(self, x, y, length, orientation, movable):
        self.x = x
        self.y = y
        self.length = length
        self.orientation = orientation
        self.movable = movable

    def get_positions(self):
        """차량의 현재 위치 반환"""
        positions = []
        if self.orientation == 'H':  # 가로 방향
            positions = [(self.x, self.y + i) for i in range(self.length)]
        elif self.orientation == 'V':  # 세로 방향
            positions = [(self.x + i, self.y) for i in range(self.length)]
        return positions


class RushHourEnv(gym.Env):
    """Rush Hour 환경 정의"""
    def __init__(self, vehicle_data, grid_size=(8, 8), max_steps=100):
        super(RushHourEnv, self).__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps = 0
        self.vehicle_data = vehicle_data

        # 차량 초기화
        self.vehicles = [
            Vehicle(data["x"], data["y"], data["length"], data["orientation"], data["movable"])
            for data in self.vehicle_data
        ]
        self.target_vehicle = self.vehicles[0]  # 첫 번째 차량을 목표 차량으로 설정
        self.exit = (3, grid_size[1] - 1)  # 출구 위치 (행 3, 마지막 열)
        self.walls = self.create_walls()  # 격자의 벽 생성

        # 행동 공간 및 관찰 공간 정의
        self.action_space = spaces.MultiDiscrete([len(self.vehicles), 2])  # (차량 번호, 이동 방향)
        self.observation_space = spaces.Box(low=0, high=1, shape=grid_size, dtype=np.int32)

        # 렌더링 초기화
        self.window_size = 400
        self.cell_size = self.window_size // self.grid_size[0]
        self.screen = None

    def create_walls(self):
        """격자 가장자리에 벽 생성"""
        walls = set()
        for i in range(self.grid_size[0]):
            walls.add((i, 0))  # 왼쪽 벽
            walls.add((i, self.grid_size[1] - 1))  # 오른쪽 벽
        for j in range(self.grid_size[1]):
            walls.add((0, j))  # 위쪽 벽
            walls.add((self.grid_size[0] - 1, j))  # 아래쪽 벽
        walls.discard(self.exit)  # 출구는 벽에서 제외
        return walls

    def reset(self):
        """환경 초기화"""
        self.steps = 0
        self.vehicles = [
            Vehicle(data["x"], data["y"], data["length"], data["orientation"], data["movable"])
            for data in self.vehicle_data
        ]
        self.target_vehicle = self.vehicles[0]  # 목표 차량 재설정
        return self.get_observation()

    def get_observation(self):
        """현재 상태를 격자로 반환"""
        grid = np.zeros(self.grid_size, dtype=np.int32)
        for i, vehicle in enumerate(self.vehicles, start=1):
            for pos in vehicle.get_positions():
                if 0 <= pos[0] < self.grid_size[0] and 0 <= pos[1] < self.grid_size[1]:
                    grid[pos] = i
        for wall in self.walls:
            grid[wall] = -1  # 벽을 -1로 표시
        grid[self.exit] = 99  # 출구 위치를 99로 표시
        return grid

    def step(self, action):
        """환경에서 한 스텝 진행"""
        vehicle_idx, direction = action
        vehicle = self.vehicles[vehicle_idx]

        if vehicle.movable and self.move_vehicle(vehicle, direction):
            reward = -1
            done = False

            # 목표 차량이 출구에 도달했는지 확인
            target_positions = self.target_vehicle.get_positions()
            if target_positions[-1] == self.exit:
                reward = 100  # 성공 보상
                done = True
            elif self.steps >= self.max_steps:
                done = True

            self.steps += 1
            return self.get_observation(), reward, done, {}
        else:
            # 이동 실패 시
            return self.get_observation(), -1, False, {}

    def move_vehicle(self, vehicle, direction):
        """차량 이동"""
        step = 1 if direction == 1 else -1
        if vehicle.orientation == 'H':  # 가로 방향
            new_y = vehicle.y + step
            new_positions = [(vehicle.x, new_y + i) for i in range(vehicle.length)]
        elif vehicle.orientation == 'V':  # 세로 방향
            new_x = vehicle.x + step
            new_positions = [(new_x + i, vehicle.y) for i in range(vehicle.length)]
        else:
            return False

        # 새 위치가 유효한지 확인
        if all(self.is_valid_position(vehicle, pos) for pos in new_positions):
            if vehicle.orientation == 'H':
                vehicle.y += step
            elif vehicle.orientation == 'V':
                vehicle.x += step
            return True
        return False

    def is_valid_position(self, vehicle, position):
        """새로운 위치가 유효한지 확인"""
        x, y = position
        if position in self.walls:  # 벽 충돌 확인
            return False
        if not (0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]):  # 그리드 경계 확인
            return False
        for other_vehicle in self.vehicles:
            if other_vehicle != vehicle and position in other_vehicle.get_positions():
                return False
        return True

    def render(self, mode='human'):
        """환경 렌더링"""
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Rush Hour Environment")

        self.screen.fill((255, 255, 255))

        for vehicle in self.vehicles:
            color = (255, 0, 0) if vehicle == self.target_vehicle else (0, 0, 255)
            for pos in vehicle.get_positions():
                rect = pygame.Rect(pos[1] * self.cell_size, pos[0] * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, color, rect)

        # 벽과 출구 렌더링
        for wall in self.walls:
            rect = pygame.Rect(wall[1] * self.cell_size, wall[0] * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (0, 0, 0), rect)
        exit_rect = pygame.Rect(self.exit[1] * self.cell_size, self.exit[0] * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (0, 255, 0), exit_rect)  # 초록색으로 출구 표시

        pygame.display.flip()

    def close(self):
        """환경 종료"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
