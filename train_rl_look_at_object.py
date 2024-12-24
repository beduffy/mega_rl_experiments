import time
from typing import Tuple, Optional
from datetime import datetime

import pybullet as p
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList

from pybullet_look_at_object_env import LookAtObjectEnv

import cProfile
import pstats
from pstats import SortKey


# TODO surely LLMs could sit beside RL algorithms and give them lots of pointer on what to do and how to change things and then train everything by RL to be 1000000 more sample efficient?


# TODO
# FPS began at 60 and went to 15 fps and then eventually segmentation fault (core dumped). Memory leak or something? lets find and fix. @train_rl_look_at_object.py @camera_controller.py  

from memory_debug import memory_tracker, profile


@profile
def make_env():
    """Create and wrap the environment"""
    memory_tracker.log_memory("Before env creation")
    env = LookAtObjectEnv(render_mode="human")
    env = DummyVecEnv([lambda: env])
    env = VecTransposeImage(env)
    memory_tracker.log_memory("After env creation")
    return env



@profile
def train():
    """Train the PPO agent"""
    # Create environment
    env = make_env()
    
    # Create unique run name with timestamp
    run_name = f"PPO_Camera_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # 0-255 confirmed
    # If you are using images as input, the observation must be of type np.uint8 
    # and be contained in [0, 255]. By default, 
    # the observation is normalized by SB3 pre-processing (dividing by 255 to have values in [0, 1]) when using CNN policies. Images can be either channel-first or channel-last.
    # If you want to use CnnPolicy or MultiInputPolicy with image-like observation (3D tensor) that are already normalized, you must pass normalize_images=False to the policy (using policy_kwargs parameter, policy_kwargs=dict(normalize_images=False)) and make sure your image is in the channel-first format.

    # Create the PPO agent
    memory_tracker.log_memory("Before model creation")
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
        ),
        # Add these parameters
        # ent_coef=0.01,          # Add entropy for exploration
        # vf_coef=0.5,            # Value function coefficient
    )
    memory_tracker.log_memory("After model creation")

    
    # Load the model and set its environment
    # model = PPO.load(
    #     "./ppo_camera_checkpoints/PPO_Camera_20241220_000145/camera_model_10000_steps",
    #     "./ppo_camera_checkpoints/PPO_Camera_20241220_000145/camera_model_10000_steps",
    #     env=env,  # Add this line
    #     print_system_info=True
    # )

    # Create a custom callback to log additional metrics
    class TensorboardCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            # self.episode_rewards = []
            # self.episode_lengths = []
            self.episode_step_counter = 0
        
        def _on_step(self):
            self.episode_step_counter += 1
            
            if self.locals.get('done'):
                # Log metrics directly without storing in lists
                episode_reward = self.locals['rewards'][0]
                episode_length = self.episode_step_counter
                self.logger.record('custom/episode_reward', episode_reward)
                self.logger.record('custom/episode_length', episode_length)
                if 'distance' in self.locals['infos'][0]:
                    self.logger.record('env/target_distance', self.locals['infos'][0]['distance'])
                self.episode_step_counter = 0
                
                # Force garbage collection after each episode
                import gc
                gc.collect()
            return True
    
    # Combine callbacks
    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=10000,
            save_path=f"./ppo_camera_checkpoints/{run_name}/",
            name_prefix="camera_model",
            verbose=1
        ),
        # TensorboardCallback()
    ])
    
    # Train the agent
    total_timesteps = 1_000_000
    # total_timesteps = 10_000
    # total_timesteps = 3_000

    memory_tracker.log_memory("Before training loop")

    # model.learn(
    #     total_timesteps=total_timesteps,
    #     callback=callbacks,
    #     progress_bar=True,
    #     log_interval=1  # Log every step
    # )

    num_steps_per_learn = 1000
    for i in range(0, total_timesteps, num_steps_per_learn):
        model.learn(
            total_timesteps=num_steps_per_learn,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=False
        )
        memory_tracker.log_memory(f"After {i+num_steps_per_learn} steps")
        
        # Force garbage collection
        import gc
        gc.collect()
        
            # Log PyBullet stats
        if p.isConnected():
            print(f"PyBullet bodies: {p.getNumBodies()}")
            print(f"PyBullet constraints: {p.getNumConstraints()}")
    
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


def profile_training():
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        train()  # Your existing train function
    finally:
        profiler.disable()
        # Output the profiling results
        stats = pstats.Stats(profiler)
        stats.sort_stats(SortKey.CUMULATIVE)
        stats.dump_stats("training_profile.prof")  # For visualization
        stats.print_stats(50)  # Print top 50 functions by time


if __name__ == "__main__":
    # from stable_baselines3.common.env_checker import check_env
    # output = check_env(make_env())
    # print('output:', output)

    profile_training()


    # evaluate("ppo_camera_final")