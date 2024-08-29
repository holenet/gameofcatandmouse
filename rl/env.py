from typing import Any, SupportsFloat
import math
from collections import deque

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces, ObservationWrapper

class CatMouseEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60, "max_steps": 1000, "catch_r": 0.1}

    def __init__(self, render_mode=None, speed_factor=4.0) -> None:
        self.speed_factor = speed_factor
        self.window_size = 1024

        self.observation_space = spaces.Dict(
            {
                "mouse_radius": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
                "cat_rotation_relative": spaces.Box(-np.pi, np.pi, shape=(1,), dtype=np.float32),
            }
        )

        self.action_space = spaces.Box(-np.pi, np.pi, shape=(1,), dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None
        self.last_mouse_coords = deque()

    def _get_obs(self):
        mouse_rotation = self.get_mouse_rotation()
        return {
            "mouse_radius": self.get_mouse_radius(),
            "cat_rotation_relative": self.normalize_rotation(self._cat_rotation - mouse_rotation),
        }
    
    def _get_info(self):
        return {}
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed, options=options)
    
        self._steps = 0
        self._mouse_coord = np.array([0.0, self.np_random.uniform(0, 0.99, (1,))[0]])
        self._cat_rotation = np.float32(self.np_random.uniform(-np.pi, np.pi, (1,))[0])
        self._progress = 0

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == 'human':
            self.last_mouse_coords.clear()
            self.last_mouse_coords.append(self._mouse_coord.copy())
            self._render_frame()
        
        return obs, info
    
    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        self._steps += 1
        prev_progress = self._progress

        mouse_speed = 0.5
        cat_speed = mouse_speed * self.speed_factor
        delta_time = 1 / self.metadata["render_fps"]

        mouse_rotation = self.get_mouse_rotation()
        delta_rotation = 0 if math.isnan(action[0]) else action[0]
        absolute_direction_rotation = mouse_rotation + delta_rotation
        self._mouse_coord += np.array([np.cos(absolute_direction_rotation), np.sin(absolute_direction_rotation)]) * mouse_speed * delta_time

        between_angle = self.normalize_rotation(self.get_mouse_rotation() - self._cat_rotation)
        if abs(between_angle) > 0.000001:
            d = np.sign(between_angle) * cat_speed * delta_time
            if abs(between_angle) < abs(d):
                d = between_angle
            self._cat_rotation = self.normalize_rotation(self._cat_rotation + d)

        if self.get_mouse_radius() < 1 / self.speed_factor:
            self._progress = abs(between_angle) / np.pi * (1 + self.get_mouse_radius() ** 2) * 0.01
        else:
            self._progress = (((abs(between_angle) - self.metadata["catch_r"]) / self.speed_factor) - (1 - self.get_mouse_radius())) * (1 + self.get_mouse_radius() ** 2)
        # print('self._progress', self._progress)
        # import time
        # time.sleep(0.2)

        if self._steps > self.metadata["max_steps"]:
            # pass
            terminated = True
            reward = -2.1
        else:
            terminated = self.get_distance() < 0.1 * self.metadata["catch_r"] or self.get_mouse_radius() >= 1
            
            if self.get_mouse_radius() >= 1:
                reward = 2 * math.tanh((abs(between_angle) - self.metadata["catch_r"]) * 20)
                
                # if self.get_distance() < self.metadata["catch_r"]:
                #     reward = -2 + self.get_distance() * 2
                # else:
                #     reward = 2 + self.get_distance() * 2
            else:
                cat_remain_dist = max(0, (abs(between_angle) - 0.1) / self.speed_factor)
                mouse_remain_dist = 1 - self.get_mouse_radius()
                # reward = -0.00001 #-0.001 #+ 0.001 * self.get_mouse_radius() ** 3 * (cat_remain_dist - mouse_remain_dist) * self.speed_factor
#                print(reward)
                reward = 0
                # if self.get_distance() < self.metadata["catch_r"]:
                    # reward = -((self.metadata["catch_r"] - self.get_distance()) / self.metadata["catch_r"]) ** 2
                # else:
                    # reward = -0.00001
                    # reward = 0
                    # reward = self._progress# - prev_progress# if self._progress - prev_progress > 0 else (self._progress - prev_progress) * 10
                    # reward = abs(between_angle) * self.get_mouse_radius() * 0.1 if self.get_mouse_radius() < 1 / self.speed_factor else self.get_mouse_radius()
                    # reward = -0.002 #+ 0.001 * self.get_mouse_radius() + 0.001 * self.get_distance()

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.last_mouse_coords.append(self._mouse_coord.copy())
            if len(self.last_mouse_coords) > 1000:
                self.last_mouse_coords.popleft()
            self._render_frame()

        return obs, reward, terminated, False, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self):
        if self.window is None and self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == 'human':
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        padding = self.window_size / 20
        pond_r = self.window_size / 2 - padding

        # pond
        pygame.draw.circle(
            canvas,
            (225, 245, 255),
            (pond_r + padding, pond_r + padding),
            pond_r
        )
        # rotation boundary
        rot_r = pond_r / self.speed_factor
        pygame.draw.arc(
            canvas,
            (255, 225, 220),
            [self.window_size / 2 - rot_r, self.window_size / 2 - rot_r, rot_r * 2, rot_r * 2],
            0, 360,
        )
        # dash boundary
        dash_r = pond_r * (1 - np.pi / self.speed_factor)
        pygame.draw.arc(
            canvas,
            (165, 160, 255),
            [self.window_size / 2 - dash_r, self.window_size / 2 - dash_r, dash_r * 2, dash_r * 2],
            0, 360,
        )
        # cat boundary
        cat_area_r = pond_r * self.metadata["catch_r"]
        cat_pos = ((self.get_cat_coord() + 1) * pond_r + padding).tolist()
        pygame.draw.arc(
            canvas,
            (0, 0, 255) if self.get_distance() < self.metadata["catch_r"] else (255, 100, 100),
            [cat_pos[0] - cat_area_r, cat_pos[1] - cat_area_r, cat_area_r * 2, cat_area_r * 2],
            0, 360,
        )
        # mouse center line
        pygame.draw.line(
            canvas,
            (200, 200, 200),
            (self.window_size / 2, self.window_size / 2),
            ((self._mouse_coord + 1) * pond_r + padding).tolist(),
        )
        # cat center line
        pygame.draw.line(
            canvas,
            (255, 200, 200),
            (self.window_size / 2, self.window_size / 2),
            cat_pos,
        )
        # mouse
        pygame.draw.circle(
            canvas,
            (150, 150, 150),
            ((self._mouse_coord + 1) * pond_r + padding).tolist(),
            pond_r / 50
        )
        # cat
        pygame.draw.circle(
            canvas,
            (0, 0, 255) if self.get_distance() < self.metadata["catch_r"] else (255, 50, 50),
            cat_pos,
            pond_r / 50
        )
        # mouse trails
        if len(self.last_mouse_coords) >= 2:
            pygame.draw.lines(
                canvas,
                (50, 255, 100),
                False,
                [((coord + 1) * pond_r + padding).tolist() for coord in self.last_mouse_coords],
            )

        if self.render_mode == 'human':
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata['render_fps'])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def get_mouse_radius(self):
        return np.linalg.norm(self._mouse_coord)

    def get_mouse_rotation(self):
        return math.atan2(self._mouse_coord[1], self._mouse_coord[0])
    
    def get_cat_coord(self):
        return np.array([math.cos(self._cat_rotation), math.sin(self._cat_rotation)]) 
    
    def get_distance(self):
        cat_coord = self.get_cat_coord()
        return np.linalg.norm(self._mouse_coord - cat_coord)

    @staticmethod
    def normalize_rotation(angle: np.float32):
        while angle <= -np.pi:
            angle += np.pi * 2
        while angle >= np.pi:
            angle -= np.pi * 2
        return angle


class CatMouseWrapper(ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Box(shape=(2,), low=0, high=1)
    
    def observation(self, observation: Any) -> Any:
        return ((observation['cat_rotation_relative'] / np.pi + 1) / 2, observation['mouse_radius'])


if __name__ == '__main__':
    env = CatMouseEnv(render_mode='human', speed_factor=0.1)
    env.reset()

    step = 0
    while True:
        step += 1
        action = env.action_space.sample()  # agent policy that uses the observation and info
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f'Terminated after {step} steps, reward is {reward}')
            obs, info = env.reset()
            step = 0

    env.close()