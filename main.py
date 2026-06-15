import time
# Hier importeren we de klassen uit de andere bestanden
from room import Room
from robot import Robot

if __name__ == "__main__":
    room_width = 12
    room_height = 12
    
    my_room = Room(width=room_width, height=room_height)
    my_robot = Robot(start_x=2, start_y=2)

    print("--- INITIAL ROOM LAYOUT ---")
    my_room.display(my_robot.x, my_robot.y)

    #robot begint (nu nog random bewegingen met de choose action)
    for step in range(20):
            chosen_action = my_robot.choose_action()
            
            # 1. Calculate tentative next position (where WOULD the robot go?)
            next_x, next_y = my_robot.x, my_robot.y
            if chosen_action == "NORTH": next_y -= 1
            elif chosen_action == "SOUTH": next_y += 1
            elif chosen_action == "EAST":  next_x += 1
            elif chosen_action == "WEST":  next_x -= 1
            
            # 2. Ask the room if this next position is allowed
            if my_room.is_move_allowed(next_x, next_y):
                print(f"Step {step + 1}: Robot moves {chosen_action} to ({next_x}, {next_y})")
                # Actual movement update
                my_robot.x, my_robot.y = next_x, next_y
            else:
                print(f"Step {step + 1}: Robot tried {chosen_action} but COLLIDED! Remaining at ({my_robot.x}, {my_robot.y})")
            
            # 3. Display the result
            my_room.display(my_robot.x, my_robot.y)
            time.sleep(1.0)