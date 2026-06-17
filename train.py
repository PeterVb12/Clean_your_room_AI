import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from robot_env import RobotTrainingEnv


# 1. Define the CNN tailored specifically for your 18x24x2 layout
class CustomGridCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        # observation_space.shape is (18, 24, 2)
        n_input_channels = observation_space.shape[-1]  # This is 2 channels

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate the flattened output size dynamically
        with th.no_grad():
            height, width = observation_space.shape[0], observation_space.shape[1]
            # Simulate a channel-first tensor format to get the output dimensions
            sample_tensor = th.zeros((1, n_input_channels, height, width))
            n_flatten = self.cnn(sample_tensor).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # FIX: Convert (Batch, Height, Width, Channels) -> (Batch, Channels, Height, Width)
        # Also cast to .float() since the environment outputs uint8 elements
        obs_permuted = observations.permute(0, 3, 1, 2).float()
        return self.linear(self.cnn(obs_permuted))


# 2. Setup vectorized training environments
env = make_vec_env(RobotTrainingEnv, n_envs=8)

# Inject our custom network configuration
policy_kwargs = dict(
    features_extractor_class=CustomGridCNN,
    features_extractor_kwargs=dict(features_dim=256),
)

print("Training generalized spatial CNN cleaning agent on dynamic room layouts...")
model = PPO(
    "MlpPolicy",  # Overridden safely by our custom CNN extractor layout
    env,
    n_steps=1024,
    batch_size=64,
    ent_coef=0.04,
    learning_rate=0.00015,
    policy_kwargs=policy_kwargs,
    verbose=1
)

# 3. Setup evaluation monitoring
eval_env = RobotTrainingEnv()
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_model",
    eval_freq=10000,
    n_eval_episodes=5,
    deterministic=True
)

# 4. Train the model
model.learn(total_timesteps=2000000, callback=eval_callback)
model.save("ppo_robot_dynamic_final")