from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.buffers import RolloutBuffer
from robot_env import RobotTrainingEnv

env = make_vec_env(RobotTrainingEnv, n_envs=8)

print("Training generalized cleaning agent on dynamic room layouts...")

# 2. Load your rock-solid baseline weights from v3
model = PPO.load("mlp_cleaner_robot_v3")
model.set_env(env)

model.n_steps = 1024
model.batch_size = 64
model.ent_coef = 0.005
model.learning_rate = 0.00015

# 4. Rebuild the rollout buffer to secure memory configurations
model.rollout_buffer = RolloutBuffer(
    model.n_steps,
    model.observation_space,
    model.action_space,
    device=model.device,
    gae_lambda=model.gae_lambda,
    gamma=model.gamma,
    n_envs=model.n_envs,
)

model.learn(total_timesteps=1200000, reset_num_timesteps=False)
model.save("dynamic_cleaner_robot_final")
print("Generalized training complete!")