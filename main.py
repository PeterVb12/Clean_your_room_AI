import time
# Hier importeren we de klassen uit de andere bestanden
from room import Room
from robot import Robot

if __name__ == "__main__":
    room_width = 12
    room_height = 12
    
    # Initialiseer de objecten uit de andere modules
    my_room = Room(width=room_width, height=room_height)
    my_robot = Robot(start_x=2, start_y=2)

    print("--- INITIAL ROOM LAYOUT ---")
    my_room.display(my_robot.x, my_robot.y)

    for step in range(3):
        chosen_action = my_robot.choose_action()
        print(f"Step {step + 1}: Robot decides to move -> {chosen_action}")
        
        my_robot.move(chosen_action)
        my_room.display(my_robot.x, my_robot.y)
        time.sleep(1.5)