import gymnasium as gym
from gymnasium import spaces
import numpy as np
from room import Room
from collections import deque


class RobotTrainingEnv(gym.Env):
    def __init__(self):
        super(RobotTrainingEnv, self).__init__()
        self.metadata = {"render_modes": ["human"]}
        self.render_mode = "human"
        self.room = Room()  # 24x18

        # Actions: 0=NORTH, 1=EAST, 2=SOUTH, 3=WEST
        self.action_space = spaces.Discrete(4)
        self.action_mapping = {0: "NORTH", 1: "EAST", 2: "SOUTH", 3: "WEST"}

        # CHANGED: Flatten the 2D matrices into a 1D vector (Size: 2 * 18 * 24 = 864)
        total_features = (2 * self.room.height * self.room.width) + 1
        self.observation_space = spaces.Box(
            low=0,
            high=2,
            shape=(total_features,),
            dtype=np.uint8
        )
        self.max_steps = 600
        coverage_test_grid = np.zeros((self.room.height, self.room.width), dtype=np.uint8)
        self.valid_mask = np.zeros((self.room.height, self.room.width), dtype=bool)

        for y in range(1, self.room.height - 1):
            for x in range(1, self.room.width - 1):
                if self.room.is_move_allowed(x, y):
                    self.valid_mask[y, x] = True
                    coverage_test_grid[y - 1:y + 2, x - 1:x + 2] = 1

        # 3. Total cleanable tiles is the sum of all squares the footprint can physically brush over
        self.total_cleanable_tiles = int(np.sum(coverage_test_grid))
        print(f"Total physical tiles the 3x3 robot can reach: {self.total_cleanable_tiles}")

        # Track if the previous move was a collision
        self.last_move_failed = 0


    def _get_obs(self):
        # 1. Channel 0: Static furniture layout
        channel_obstacles = self.room.grid.copy().astype(np.uint8)

        # 2. Channel 1: Cleaning progress layer
        channel_cleaned = self.cleaned_grid.copy()

        # CHANGED: Stamp the entire 3x3 footprint of the robot onto the observation layer
        # Since the robot center is at (robot_x, robot_y), we fill from center-1 to center+1
        y_start, y_end = self.robot_y - 1, self.robot_y + 2
        x_start, x_end = self.robot_x - 1, self.robot_x + 2

        # Mark all 9 tiles of the robot's body as '2'
        channel_cleaned[y_start:y_end, x_start:x_end] = 2

        # Stack the channels together
        stacked_grid = np.stack([channel_obstacles, channel_cleaned], axis=0)
        flattened_grid = stacked_grid.flatten()

        # Append the collision flag to the end
        return np.append(flattened_grid, [self.last_move_failed]).astype(np.uint8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 1. Generate a completely randomized room layout on every reset
        self.room = Room()

        # 2. Dynamically find a valid starting position
        # We can no longer guarantee that a specific coordinate is open!
        import random
        valid_ys, valid_xs = np.where(self.valid_mask)
        while True:
            idx = random.randint(0, len(valid_xs) - 1)
            start_x = int(valid_xs[idx])
            start_y = int(valid_ys[idx])
            if self.room.is_move_allowed(start_x, start_y):
                self.robot_x = start_x
                self.robot_y = start_y
                break

        # 3. Recalculate maximum cleanable tiles for THIS specific random layout
        coverage_test_grid = np.zeros((self.room.height, self.room.width), dtype=np.uint8)
        for y in range(1, self.room.height - 1):
            for x in range(1, self.room.width - 1):
                if self.room.is_move_allowed(x, y):
                    coverage_test_grid[y - 1:y + 2, x - 1:x + 2] = 1

        self.total_cleanable_tiles = int(np.sum(coverage_test_grid))

        # 4. Clear state variables and flags
        self.current_step = 0
        self.last_move_failed = 0
        self.cleaned_grid = np.zeros((self.room.height, self.room.width), dtype=np.uint8)

        # 5. Clean the initial 3x3 spawn footprint immediately
        y_start, y_end = self.robot_y - 1, self.robot_y + 2
        x_start, x_end = self.robot_x - 1, self.robot_x + 2
        self.cleaned_grid[y_start:y_end, x_start:x_end] = 1
        self.tiles_cleaned_count = int(np.sum(self.cleaned_grid))

        return self._get_obs(), {}

    def _bfs_dirty_distances(self):
        # any cell whose 3x3 footprint still touches unfinished dirt = BFS source
        dirty = (self.room.grid == 0) & (self.cleaned_grid == 0)
        H, W = self.room.height, self.room.width
        any3 = dirty.copy()
        for sdy in (-1, 0, 1):
            for sdx in (-1, 0, 1):
                if sdx == 0 and sdy == 0:
                    continue
                shifted = np.zeros_like(dirty)
                ys, ye = max(0, sdy), H + min(0, sdy)
                xs, xe = max(0, sdx), W + min(0, sdx)
                yd, yde = max(0, -sdy), H + min(0, -sdy)
                xd, xde = max(0, -sdx), W + min(0, -sdx)
                shifted[yd:yde, xd:xde] = dirty[ys:ye, xs:xe]
                any3 |= shifted
        frontier = any3 & self.valid_mask

        dist = np.full((H, W), -1, dtype=np.int32)
        ys, xs = np.where(frontier)
        if len(ys) == 0:
            return None  # nothing dirty left, skip shaping

        q = deque(zip(xs.tolist(), ys.tolist()))
        dist[ys, xs] = 0
        while q:
            x, y = q.popleft()
            d = dist[y, x] + 1
            for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0)):
                nx, ny = x + dx, y + dy
                if 0 <= ny < H and 0 <= nx < W and self.valid_mask[ny, nx] and dist[ny, nx] == -1:
                    dist[ny, nx] = d
                    q.append((nx, ny))
        return dist

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

        reward = -0.01  # Tiny base clock penalty

        if self.room.is_move_allowed(next_x, next_y):
            # 1. Get distance before moving
            dist_grid_old = self._bfs_dirty_distances()
            old_dist = 0 if dist_grid_old is None else dist_grid_old[self.robot_y, self.robot_x]
            if old_dist == -1: old_dist = 0

            # 2. Execute move
            self.robot_x, self.robot_y = next_x, next_y
            self.last_move_failed = 0

            # 3. Process the vacuum cleaning loop over cleaned_grid here...
            # [Keep your current 3x3 loop that increments self.tiles_cleaned_count]

            # 4. Get the fresh distance grid AFTER changes to coordinates and dirt
            dist_grid_new = self._bfs_dirty_distances()
            new_dist = 0 if dist_grid_new is None else dist_grid_new[self.robot_y, self.robot_x]
            if new_dist == -1: new_dist = old_dist

            # 5. Distribute reward shaping signals
            if new_dist < old_dist:
                reward += 0.15
            elif new_dist > old_dist:
                reward -= 0.15



            # --- EVALUATE 3x3 VACUUM CLEANING ---
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
                reward += 5.0 * new_tiles_cleaned
            else:
                # ONLY apply backtracking penalty if we aren't moving towards a target
                if new_dist >= old_dist:
                    coverage_ratio = self.tiles_cleaned_count / self.total_cleanable_tiles
                    dynamic_penalty = -0.15 * (1.0 - coverage_ratio)
                    reward += min(dynamic_penalty, -0.01)

        else:
            # Handle wall collisions
            self.last_move_failed = 1
            reward += -0.25

        terminated = False
        truncated = False

        if self.tiles_cleaned_count >= self.total_cleanable_tiles:
            reward += 150.0 + 1.0 * (self.max_steps - self.current_step)  # early finish >> late finish
            terminated = True

        if self.current_step >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        # Gymnasium triggers this check automatically.
        # If no rendering is intended or supported, we catch it here.
        if self.render_mode is None:
            return

        # Hooks directly into your original ASCII map visualizer logic
        if self.render_mode == "human":
            for y in range(self.room.height):
                row_str = ""
                for x in range(self.room.width):
                    if x == self.robot_x and y == self.robot_y:
                        row_str += " R "
                    elif self.room.grid[y, x] == 1:
                        row_str += " # "
                    elif self.cleaned_grid[y, x] == 1:
                        row_str += " o "  # Cleaned path
                    else:
                        row_str += " . "  # Dirty tile
                print(row_str)
            print(f"Progress: {self.tiles_cleaned_count}/{self.total_cleanable_tiles} tiles cleaned.")
            print("\n" + "-" * 30 + "\n")