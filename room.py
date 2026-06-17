import numpy as np
import random

class Room:
    def __init__(self, width=24, height=18):
        self.width = width
        self.height = height
        self.grid = np.zeros((self.height, self.width), dtype=int)
        self._generate_solvable_room()

    def _generate_solvable_room(self, max_attempts=50):
        for _ in range(max_attempts):
            self.grid = np.zeros((self.height, self.width), dtype=int)
            self._generate_obstacles()
            if self._is_fully_solvable():
                return
        self._generate_hardcoded_room()

    def _is_fully_solvable(self, start_x=1, start_y=1):
        """
        Genuinely evaluates rooms for a 3x3 robot. Maps out the exact physical
        footprint of all reachable center coordinates. If any empty tile is
        left unreached (blindspots/gaps), it rejects the room.
        """
        if not self.is_move_allowed(start_x, start_y):
            return False

        # 1. Flood-fill (BFS) to find all reachable center coordinates from spawn
        seen = {(start_x, start_y)}
        queue = [(start_x, start_y)]
        head = 0
        while head < len(queue):
            x, y = queue[head]
            head += 1
            for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0)):
                nx, ny = x + dx, y + dy
                if (nx, ny) not in seen and self.is_move_allowed(nx, ny):
                    seen.add((nx, ny))
                    queue.append((nx, ny))

        # 2. Build a coverage grid representing what the robot can actually sweep
        accessible_vacuum_zone = np.zeros((self.height, self.width), dtype=np.uint8)
        for (cx, cy) in seen:
            accessible_vacuum_zone[cy-1:cy+2, cx-1:cx+2] = 1

        # 3. Check for streaks or unreachable floor tiles.
        # Every single open tile (grid == 0) MUST be covered by the vacuum zone.
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] == 0 and accessible_vacuum_zone[y, x] == 0:
                    return False  # Leaves behind an unfinishable dirty spot!

        return True
    def _generate_obstacles(self):
        furniture_templates = [(4, 8), (10, 3), (6, 4)]
        placed_obstacles = []  # To track coords for printing

        for i, (item_w, item_h) in enumerate(furniture_templates):
            placed = False
            attempts = 0

            if random.choice([True, False]):
                item_w, item_h = item_h, item_w

            while not placed and attempts < 100:
                attempts += 1

                if i < 2:
                    wall = random.randint(0, 3)
                    if wall == 0:
                        x = random.randint(0, self.width - item_w)
                        y = 0
                    elif wall == 1:
                        x = random.randint(0, self.width - item_w)
                        y = self.height - item_h
                    elif wall == 2:
                        x = 0
                        y = random.randint(0, self.height - item_h)
                    else:
                        x = self.width - item_w
                        y = random.randint(0, self.height - item_h)
                else:
                    x = random.randint(0, self.width - item_w)
                    y = random.randint(0, self.height - item_h)

                if x < 0 or y < 0 or x + item_w > self.width or y + item_h > self.height:
                    continue

                if np.all(self.grid[y:y + item_h, x:x + item_w] == 0):
                    self.grid[y:y + item_h, x:x + item_w] = 1
                    placed_obstacles.append({
                        "item_index": i,
                        "top_left": (x, y),
                        "width": item_w,
                        "height": item_h
                    })
                    placed = True

    def _generate_hardcoded_room(self):
        self.width,self.height = 24,18
        self.grid = np.zeros((self.height, self.width), dtype=int)
        fixed_obstacles = [(20, 6, 4, 8),(6, 12, 10, 3), (0, 4, 6, 4)]
        for x, y, w, h in fixed_obstacles: self.grid[y:y + h, x:x + w] = 1


    def display(self, robot_x, robot_y):
        for y in range(self.height):
            row_str = ""
            for x in range(self.width):
                if x == robot_x and y == robot_y:
                    row_str += " R "
                elif abs(x - robot_x) <= 1 and abs(y - robot_y) <= 1:
                    row_str += " r "
                elif self.grid[y, x] == 1:
                    row_str += " # "
                elif self.grid[y, x] == 2:
                    row_str += " o "
                else:
                    row_str += " . "
            print(row_str)
        print("\n" + "-" * 30 + "\n")
    
    def is_move_allowed(self, new_x, new_y):
        """
        Checks if a 3x3 robot centered at (new_x, new_y) fits within the boundaries
        and does not collide with any obstacles (1).
        """
        # 1. Boundary check (Muren)
        # Since the robot is 3x3, the center must keep a margin of 1 from all edges
        if new_x < 1 or new_x > self.width - 2:
            return False
        if new_y < 1 or new_y > self.height - 2:
            return False

        # 2. Obstacle check (Meubels)
        # Slice a 3x3 area from the grid centered around the new position
        # Remember: grid is indexed as [y, x]!
        y_start, y_end = new_y - 1, new_y + 2
        x_start, x_end = new_x - 1, new_x + 2
        robot_area = self.grid[y_start:y_end, x_start:x_end]

        # If there is a '1' anywhere in this 3x3 area, it's a collision!
        if np.any(robot_area == 1):
            return False

        # If both checks pass, the move is perfectly fine
        return True