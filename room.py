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