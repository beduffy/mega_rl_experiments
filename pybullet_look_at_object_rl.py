import time
from typing import Tuple, Optional

import pybullet as p
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from pybullet_look_at_object_env_vla import LookAtObjectEnv


def make_env():
    """Create and wrap the environment"""
    # env = LookAtObjectEnv()
    env = LookAtObjectEnv('robot')
    # Wrap env in Monitor to log training stats
    env = Monitor(env)
    return env


def train():
    """Train the PPO agent"""
    # Create environment
    env = make_env()
    
    # Create unique run name with timestamp
    from datetime import datetime
    run_name = f"PPO_Camera_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create the PPO agent
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log=f"./ppo_camera_tensorboard/{run_name}",
        # Add policy network configuration
        policy_kwargs=dict(
            net_arch=dict(
                pi=[64, 64],  # Policy network
                vf=[64, 64]   # Value function network
            )
        )
    )
    
    # Create a custom callback to log additional metrics
    from stable_baselines3.common.callbacks import BaseCallback
    
    class TensorboardCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.episode_rewards = []
            self.episode_lengths = []
        
        def _on_step(self):
            if self.locals.get('done'):
                # Log episode metrics
                self.logger.record('custom/episode_reward', self.locals['rewards'][0])
                self.logger.record('custom/episode_length', self.locals['dones'].sum())
                # Log environment info
                if 'distance' in self.locals['infos'][0]:
                    self.logger.record('env/target_distance', self.locals['infos'][0]['distance'])
            return True
    
    # Combine callbacks
    from stable_baselines3.common.callbacks import CallbackList
    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=100,
            save_path=f"./ppo_camera_checkpoints/{run_name}/",
            name_prefix="camera_model",
            verbose=1
        ),
        TensorboardCallback()
    ])
    
    # Train the agent
    total_timesteps = 1_000_000
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
        log_interval=1  # Log every step
    )
    
    # Save the final model
    model.save(f"./ppo_camera_checkpoints/{run_name}/final_model")


def evaluate(model_path: str, num_episodes: int = 10):
    """Evaluate a trained model"""
    env = make_env()
    model = PPO.load(model_path)
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            p.stepSimulation()
            time.sleep(1./240.)
        
        print(f"Episode {episode + 1}: Reward = {episode_reward}")
    
    env.close()


if __name__ == "__main__":
    # Train the agent
    train()
    
    # Evaluate the trained model
    evaluate("ppo_camera_final")