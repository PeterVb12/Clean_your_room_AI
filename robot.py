import random

class Robot:
    def __init__(self, start_x=1, start_y=1):
        self.x = start_x
        self.y = start_y
        self.q_table = {}

    def choose_action(self):
        actions = ["NORTH", "EAST", "SOUTH", "WEST"]
        return random.choice(actions)

    def move(self, action):
        if action == "NORTH": self.y -= 1
        elif action == "SOUTH": self.y += 1
        elif action == "EAST":  self.x += 1
        elif action == "WEST":  self.x -= 1