import gym
import numpy as np
import pygame
from gym import spaces
from collections import deque
import numpy as np
import pygame
from gym import spaces

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
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps = 0
        self.window_size = 400
        self.cell_size = self.window_size // self.grid_size[0]
        self.screen = None

        # Initialize vehicles
        self.vehicle_data = vehicle_data
        self.vehicles = [
            Vehicle(data["x"], data["y"], data["length"], data["orientation"], data["movable"])
            for data in self.vehicle_data
        ]
        self.target_vehicle = self.vehicles[0]  # First vehicle is the target vehicle
        self.exit_wall = (3, self.grid_size[1] - 1)  # Exit position
        self.walls = self.create_walls()

        self.action_space = spaces.MultiDiscrete([len(self.vehicles), 2])  # (vehicle index, direction)
        self.observation_space = spaces.Box(low=0, high=1, shape=self.grid_size, dtype=np.int32)

    def create_walls(self):
        walls = set()
        for i in range(self.grid_size[0]):
            walls.add((i, 0))
            walls.add((i, self.grid_size[1] - 1))
        for j in range(self.grid_size[1]):
            walls.add((0, j))
            walls.add((self.grid_size[0] - 1, j))
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
        grid = np.zeros(self.grid_size, dtype=np.int32)
        for i, vehicle in enumerate(self.vehicles, start=1):
            for pos in vehicle.get_positions():
                if 0 <= pos[0] < self.grid_size[0] and 0 <= pos[1] < self.grid_size[1]:
                    grid[pos] = i
        for wall in self.walls:
            grid[wall] = -1
        return grid

    def step(self, action):
        vehicle_idx, direction = [int(a) for a in action]
        vehicle = self.vehicles[vehicle_idx]
        previous_position = set(vehicle.get_positions())  # Store the current positions before the move

        if vehicle.movable and self.move_vehicle(vehicle, direction, previous_position):
            self.steps += 1
            reward = self.calculate_reward(vehicle, previous_position)
            done = False

            if self.target_vehicle.get_positions()[-1] == (self.exit_wall[0], self.exit_wall[1] - 1):
                reward = 100
                done = True
            elif self.steps >= self.max_steps:
                done = True

            return self.get_observation(), reward, done, {}
        else:
            return self.get_observation(), -0.5, False, {}

    def move_vehicle(self, vehicle, direction, previous_position):
        step = 1 if direction == 1 else -1
        new_positions = vehicle.get_positions()

        if vehicle.orientation == 'H':
            new_positions = [(vehicle.x, vehicle.y + step + i) for i in range(vehicle.length)]
        elif vehicle.orientation == 'V':
            new_positions = [(vehicle.x + step + i, vehicle.y) for i in range(vehicle.length)]

        can_move = all(self.is_valid_position(pos, previous_position) for pos in new_positions)
        
        if can_move:
            if vehicle.orientation == 'H':
                vehicle.y += step
            elif vehicle.orientation == 'V':
                vehicle.x += step
            return True
        return False

    def is_valid_position(self, position, previous_position):
        x, y = position
        if position in self.walls or not (0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]):
            return False
        for other_vehicle in self.vehicles:
            if position in other_vehicle.get_positions() and position not in previous_position:
                return False
        return True

    def calculate_reward(self, vehicle, previous_position):
        if self.target_vehicle.get_positions()[-1] == (self.exit_wall[0], self.exit_wall[1] - 1):
            return 100
        if set(vehicle.get_positions()) != previous_position:
            return 10  # Reward for successful move
        return -0.1  # Penalty for staying

    def render(self, mode="human"):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Rush Hour Environment")
        self.screen.fill((255, 255, 255))
        for i, vehicle in enumerate(self.vehicles, start=1):
            color = (255, 0, 0) if vehicle == self.target_vehicle else (0, 0, 255)
            x_start, y_start = vehicle.get_positions()[0]
            if vehicle.orientation == 'H':
                rect = pygame.Rect(y_start * self.cell_size, x_start * self.cell_size, self.cell_size * vehicle.length, self.cell_size)
            elif vehicle.orientation == 'V':
                rect = pygame.Rect(y_start * self.cell_size, x_start * self.cell_size, self.cell_size, self.cell_size * vehicle.length)
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, (0, 0, 0), rect, width=2)
        for wall in self.walls:
            rect = pygame.Rect(wall[1] * self.cell_size, wall[0] * self.cell_size, self.cell_size, self.cell_size)
            color = (0, 255, 0) if wall == self.exit_wall else (0, 0, 0)
            pygame.draw.rect(self.screen, color, rect)
        pygame.display.update()
