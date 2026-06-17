import numpy as np
from stable_baselines3 import PPO
from robot_env import RobotTrainingEnv

env = RobotTrainingEnv()
model = PPO.load("ppo_robot_dynamic_final")

num_test_episodes = 100
success_count = 0
all_episode_lengths = []
all_coverage_pcts = []

print(f"Evaluating the trained model over {num_test_episodes} randomized rooms...")

for episode in range(num_test_episodes):
    obs, info = env.reset()
    done = False
    steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)

        # Convert the NumPy array action into a standard Python integer
        action = action.item()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

    # Track metrics
    coverage = env.tiles_cleaned_count / env.total_cleanable_tiles
    all_coverage_pcts.append(coverage)
    all_episode_lengths.append(steps)

    if terminated:  # Hit the 95% threshold early
        success_count += 1

print("\n================ EVALUATION RESULTS ================")
print(f"Total Rooms Tested:      {num_test_episodes}")
print(f"Room Completion Rate:    {(success_count / num_test_episodes) * 100:.1f}%")
print(f"Average Final Coverage:  {np.mean(all_coverage_pcts) * 100:.1f}%")
print(f"Average Steps Per Room:  {np.mean(all_episode_lengths):.1f} steps")
print("====================================================")