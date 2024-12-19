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
    env = LookAtObjectEnv()
    # Wrap env in Monitor to log training stats
    env = Monitor(env)
    return env


def train():
    """Train the PPO agent"""
    # Create environment
    env = make_env()
    
    # Verify the environment follows gym interface
    check_env(env)
    
    # Create the PPO agent
    model = PPO(
        "CnnPolicy",  # Use CNN policy since we have image observations
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="./ppo_camera_tensorboard/"
    )
    
    # Setup automatic checkpointing
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./ppo_camera_checkpoints/",
        name_prefix="camera_model"
    )
    
    # Train the agent
    total_timesteps = 1_000_000
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save the final model
    model.save("ppo_camera_final")


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