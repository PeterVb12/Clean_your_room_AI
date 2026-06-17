import gymnasium as gym
from gymnasium import spaces
import numpy as np
from room import Room


class RobotTrainingEnv(gym.Env):
    def __init__(self):
        super(RobotTrainingEnv, self).__init__()
        self.metadata = {"render_modes": ["human"]}
        self.render_mode = "human"
        self._new_room()

        # Actions: 0=NORTH, 1=EAST, 2=SOUTH, 3=WEST
        self.action_space = spaces.Discrete(4)
        self.action_mapping = {0: "NORTH", 1: "EAST", 2: "SOUTH", 3: "WEST"}

        # UPGRADE: Shape is now (18, 24, 3) to pass the navigation map
        self.observation_space = spaces.Box(
            low=0,
            high=3,
            shape=(self.room.height, self.room.width, 3),
            dtype=np.uint8
        )
        self.max_steps = 800

    def _get_obs(self):
        # Channel 0: Static furniture layouts
        channel_obstacles = self.room.grid.copy().astype(np.uint8)

        # Channel 1: Cleaning progress layer
        channel_cleaned = self.cleaned_grid.copy()

        # Stamp the 3x3 footprint of the robot onto the observation layer
        y_start, y_end = self.robot_y - 1, self.robot_y + 2
        x_start, x_end = self.robot_x - 1, self.robot_x + 2
        channel_cleaned[y_start:y_end, x_start:x_end] = 2

        # UPGRADE: Explicitly mark the center pixel as '3' so the robot tracks its pivot point
        channel_cleaned[self.robot_y, self.robot_x] = 3

        # UPGRADE: Channel 2 is a binary navigation grid (1 = Allowed Center, 0 = Forbidden/Collision)
        channel_navigation = self.valid_mask.astype(np.uint8)

        # Stack along the last axis to make it an (18, 24, 3) image matrix
        stacked_grid = np.stack([channel_obstacles, channel_cleaned, channel_navigation], axis=-1)
        return stacked_grid.astype(np.uint8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._new_room()

        self.robot_x = 1
        self.robot_y = 1

        self.current_step = 0
        self.last_move_failed = 0
        self.cleaned_grid = np.zeros((self.room.height, self.room.width), dtype=np.uint8)

        # Clean the initial 3x3 spawn footprint
        y_start, y_end = self.robot_y - 1, self.robot_y + 2
        x_start, x_end = self.robot_x - 1, self.robot_x + 2
        self.cleaned_grid[y_start:y_end, x_start:x_end] = 1
        self.tiles_cleaned_count = int(np.sum(self.cleaned_grid))

        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        direction = self.action_mapping[action]

        next_x, next_y = self.robot_x, self.robot_y
        if direction == "NORTH":
            next_y -= 1
        elif direction == "SOUTH":
            next_y += 1
        elif direction == "EAST":
            next_x += 1
        elif direction == "WEST":
            next_x -= 1

        reward = -0.05  # Timestep penalty

        if self.room.is_move_allowed(next_x, next_y):
            self.robot_x, self.robot_y = next_x, next_y
            self.last_move_failed = 0

            # Vacuum loop
            y_start, y_end = self.robot_y - 1, self.robot_y + 2
            x_start, x_end = self.robot_x - 1, self.robot_x + 2

            new_tiles_cleaned = 0
            for cy in range(y_start, y_end):
                for cx in range(x_start, x_end):
                    if self.cleaned_grid[cy, cx] == 0:
                        self.cleaned_grid[cy, cx] = 1
                        new_tiles_cleaned += 1
                        self.tiles_cleaned_count += 1

            if new_tiles_cleaned > 0:
                reward += 1.0 * new_tiles_cleaned
        else:
            self.last_move_failed = 1
            reward += -0.25  # Collision penalty

        coverage_ratio = self.tiles_cleaned_count / self.total_cleanable_tiles

        terminated = False
        truncated = False

        if coverage_ratio >= 0.95:
            time_bonus = 0.1 * (self.max_steps - self.current_step)
            reward += 100.0 + time_bonus
            terminated = True

        if self.current_step >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode is None:
            return

        if self.render_mode == "human":
            for y in range(self.room.height):
                row_str = ""
                for x in range(self.room.width):
                    if x == self.robot_x and y == self.robot_y:
                        row_str += " R "
                    elif self.room.grid[y, x] == 1:
                        row_str += " # "
                    elif self.cleaned_grid[y, x] == 1:
                        row_str += " o "
                    else:
                        row_str += " . "
                print(row_str)
            print(f"Progress: {self.tiles_cleaned_count}/{self.total_cleanable_tiles} tiles cleaned.")
            print("\n" + "-" * 30 + "\n")

    def _new_room(self):
        self.room = Room()
        coverage_test_grid = np.zeros((self.room.height, self.room.width), dtype=np.uint8)
        self.valid_mask = np.zeros((self.room.height, self.room.width), dtype=bool)

        # Calculate valid masks and target metrics using a real reachability pass from (1,1)
        seen = {(1, 1)}
        queue = [(1, 1)]
        head = 0
        while head < len(queue):
            x, y = queue[head]
            head += 1
            for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0)):
                nx, ny = x + dx, y + dy
                if (nx, ny) not in seen and self.room.is_move_allowed(nx, ny):
                    seen.add((nx, ny))
                    queue.append((nx, ny))

        for (cx, cy) in seen:
            self.valid_mask[cy, cx] = True
            coverage_test_grid[cy - 1:cy + 2, cx - 1:cx + 2] = 1

        self.total_cleanable_tiles = int(np.sum(coverage_test_grid))