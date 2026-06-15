import numpy as np
import random

class Room:
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.grid = np.zeros((self.height, self.width), dtype=int)
        self._generate_obstacles()

    def _generate_obstacles(self):
        furniture_templates = [(2, 2), (1, 3), (2, 1)]
        for item_w, item_h in furniture_templates:
            placed = False
            attempts = 0
            while not placed and attempts < 100:
                attempts += 1
                if random.choice([True, False]):
                    item_w, item_h = item_h, item_w
                x = random.randint(0, self.width - item_w)
                y = random.randint(0, self.height - item_h)
                
                if np.all(self.grid[y:y+item_h, x:x+item_w] == 0):
                    self.grid[y:y+item_h, x:x+item_w] = 1
                    placed = True

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