import time
from stable_baselines3 import PPO
from robot_env import RobotTrainingEnv

# 1. Instantiate the updated environment
env = RobotTrainingEnv()
model = PPO.load("mlp_cleaner_robot_v3")
#mlp_cleaner_robot_v3
#dynamic_cleaner_robot_final

obs, info = env.reset()
done = False
steps = 0

print("\n--- AI AGENT RUNNING ---")
env.render()  # This will now call your ASCII display smoothly
time.sleep(1.0)

# Track both termination and truncation caps
while not done and steps < 1200:
    # 2. Get action from the model
    action, _states = model.predict(obs, deterministic=False)

    # 3. Take the step (cast action to int)
    obs, reward, terminated, truncated, info = env.step(int(action))

    print(f"Step {steps + 1}: Robot moved. Reward earned: {reward:.2f}")
    env.render()

    done = terminated or truncated
    steps += 1
    time.sleep(0.1)  # Lowered this slightly so you can watch it clean at a steady pace

if terminated:
    print(f"Success! Room completely cleaned in {steps} steps.")
else:
    print("Time ran out before 100% completion.")