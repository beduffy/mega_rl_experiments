import pygame
import numpy as np
import h5py
import time
import os
from dataclasses import dataclass
from typing import Tuple, Optional
import argparse
import torch
from imitate_1_episode_point_at_env import SimplePolicy  # Import our policy class


@dataclass
class ServoState:
    """Current state of the servo system"""
    current_angle: float  # Current angle in radians
    target_angle: float  # Target angle in radians
    velocity: float      # Current angular velocity


class ServoEnv:
    """
    Simulated servo environment with a flag pointing at a target.
    
    State space:
        - current_angle: float [-π, π]
        - angular_velocity: float
    Action space:
        - target_angle: float [-π, π]
    
    Observation space:
        - RGB image (240, 240, 3)
        - servo_state (2,) [current_angle, angular_velocity]
    """
    
    def __init__(self, 
                 screen_size: int = 240,
                 servo_radius: int = 30,
                 flag_length: int = 40,
                 target_radius: int = 10,
                 max_velocity: float = 5.0,
                 kp: float = 3.0,
                 kd: float = 0.5,
                 target_threshold: float = 0.01,
                 max_steps: int = 200):
        
        # Initialize pygame
        pygame.init()
        self.screen_size = screen_size
        self.screen = pygame.display.set_mode((screen_size, screen_size))
        pygame.display.set_caption("Servo Flag Environment")
        
        # Environment parameters
        self.servo_radius = servo_radius
        self.flag_length = flag_length
        self.target_radius = target_radius
        self.servo_center = (screen_size // 2, screen_size // 2)
        self.max_velocity = max_velocity
        
        # PD controller parameters
        self.kp = kp
        self.kd = kd
        
        self.target_threshold = target_threshold
        
        # Colors - Move these before reset()
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        
        # Recording variables
        self.recording = False
        self.recorded_data = {
            'images': [],
            'servo_states': [],
            'actions': [],
            'rewards': [],
            'timestamps': []
        }
        
        self.max_steps = max_steps
        self.steps = 0  # Initialize step counter
        self.reset()


    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.state = ServoState(
            current_angle=np.random.uniform(-np.pi, np.pi),
            target_angle=0.0,
            velocity=0.0
        )
        self.target_pos = self._sample_target_position()
        self.steps = 0  # Reset step counter
        return self._get_observation()
    

    def _sample_target_position(self) -> Tuple[int, int]:
        """Sample a random target position (excluding center region)"""
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(self.servo_radius + self.flag_length,
                                   self.screen_size // 2 - self.target_radius)
        x = self.servo_center[0] + int(distance * np.cos(angle))
        y = self.servo_center[1] + int(distance * np.sin(angle))
        return (x, y)


    def _get_observation(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return current observation (image and servo state)"""
        # Render and get pixel array
        self._render()
        image = pygame.surfarray.array3d(self.screen)
        image = np.transpose(image, (1, 0, 2))  # Convert to (H, W, C)
        
        # Get servo state
        servo_state = np.array([self.state.current_angle, self.state.velocity])
        
        return image, servo_state


    def _compute_reward(self) -> float:
        """Compute reward based on flag alignment with target"""
        # Calculate angle to target
        dx = self.target_pos[0] - self.servo_center[0]
        dy = self.target_pos[1] - self.servo_center[1]
        target_angle = np.arctan2(dy, dx)
        
        # Calculate angular difference (normalized to [-π, π])
        angle_diff = np.abs(target_angle - self.state.current_angle)
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        
        # Reward is 1 when pointing exactly at target, decreases with angular difference
        reward = np.exp(-5 * angle_diff**2)  # Gaussian reward
        return float(reward)


    def step(self, action: float) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take a step in the environment.
        
        Args:
            action: target angle in radians [-π, π]
        Returns:
            observation: (image, servo_state)
            reward: float
            done: True if episode is finished (max steps or other conditions)
            info: dict with additional information
        """
        self.steps += 1
        self.state.target_angle = action
        
        # PD control update
        angle_diff = self.state.target_angle - self.state.current_angle
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        
        desired_velocity = self.kp * angle_diff
        velocity_diff = desired_velocity - self.state.velocity
        
        # Update velocity with PD control
        self.state.velocity += self.kd * velocity_diff
        self.state.velocity = np.clip(self.state.velocity, -self.max_velocity, self.max_velocity)
        
        # Update angle
        self.state.current_angle += self.state.velocity * 0.1  # dt = 0.1
        self.state.current_angle = (self.state.current_angle + np.pi) % (2 * np.pi) - np.pi
        
        # Check if we need to relocate target
        dx = self.target_pos[0] - self.servo_center[0]
        dy = self.target_pos[1] - self.servo_center[1]
        target_angle = np.arctan2(dy, dx)
        
        # Calculate angular difference to target
        angle_diff = np.abs(target_angle - self.state.current_angle)
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        
        # If we're close enough to target, relocate it
        if abs(angle_diff) < self.target_threshold:
            self.target_pos = self._sample_target_position()
        
        # Get observation and reward
        obs = self._get_observation()
        reward = self._compute_reward()
        
        # Check if episode should end (max steps reached)
        done = self.steps >= self.max_steps
        if done:
            self.reset()
        
        # Record if recording is enabled
        if self.recording:
            self.recorded_data['images'].append(obs[0])
            self.recorded_data['servo_states'].append(obs[1])
            self.recorded_data['actions'].append(action)
            self.recorded_data['rewards'].append(reward)
            self.recorded_data['timestamps'].append(time.time())
        
        return obs, reward, done, {'steps': self.steps}


    def _render(self) -> None:
        """Render the environment"""
        self.screen.fill(self.WHITE)
        
        # Draw target
        pygame.draw.circle(self.screen, self.RED, self.target_pos, self.target_radius)
        
        # Draw servo base
        pygame.draw.circle(self.screen, self.BLACK, self.servo_center, self.servo_radius)
        
        # Draw flag (line)
        flag_end = (
            self.servo_center[0] + int(self.flag_length * np.cos(self.state.current_angle)),
            self.servo_center[1] + int(self.flag_length * np.sin(self.state.current_angle))
        )
        pygame.draw.line(self.screen, self.BLUE, self.servo_center, flag_end, 4)
        
        pygame.display.flip()


    def start_recording(self) -> None:
        """Start recording episode data"""
        self.recording = True
        self.recorded_data = {
            'images': [],
            'servo_states': [],
            'actions': [],
            'rewards': [],
            'timestamps': []
        }


    def stop_recording(self) -> None:
        """Stop recording and save episode data"""
        self.recording = False


    def save_recording(self, filename: str) -> None:
        """Save recorded episode to HDF5 file"""
        # Convert lists to numpy arrays before saving
        images = np.array(self.recorded_data['images'])
        qpos = np.array(self.recorded_data['servo_states'])
        actions = np.array(self.recorded_data['actions'])[:, None]
        rewards = np.array(self.recorded_data['rewards'])
        timestamps = np.array(self.recorded_data['timestamps'])
        
        print(f"Saving data shapes:")
        print(f"  Images: {images.shape}")
        print(f"  QPos: {qpos.shape}")
        print(f"  Actions: {actions.shape}")
        print(f"  Rewards: {rewards.shape}")
        
        try:
            with h5py.File(filename, 'w') as f:
                f.attrs['sim'] = True
                f.attrs['num_steps'] = len(images)
                
                # Save observations
                f.create_dataset('/observations/images/main',
                               data=images,
                               compression='gzip')
                f.create_dataset('/observations/qpos',
                               data=qpos,
                               compression='gzip')
                
                # Save actions and rewards
                f.create_dataset('/action',
                               data=actions,
                               compression='gzip')
                f.create_dataset('/rewards',
                               data=rewards,
                               compression='gzip')
                f.create_dataset('/timestamps',
                               data=timestamps,
                               compression='gzip')
                
                # Ensure file is flushed
                f.flush()
            
            print(f"Successfully saved recording to {filename}")
            
        except Exception as e:
            print(f"Error saving recording: {e}")
            raise


def scripted_controller(env: ServoEnv, num_steps: int = 1000) -> None:
    """Scripted PD controller that points at target"""
    env.start_recording()
    obs = env.reset()
    
    try:
        for step in range(num_steps):
            # Calculate angle to target
            dx = env.target_pos[0] - env.servo_center[0]
            dy = env.target_pos[1] - env.servo_center[1]
            target_angle = np.arctan2(dy, dx)
            
            # Take action
            obs, reward, done, _ = env.step(target_angle)
            
            # Print every 100 steps
            if step > 0 and step % 100 == 0:
                print(f"Step {step}/{num_steps}")
            
            pygame.event.pump()
            time.sleep(0.01)
    
    finally:
        env.stop_recording()
        filename = f'scripted_episode_{num_steps}steps.hdf5'
        print(f"Saving {num_steps} steps to {filename}...")
        env.save_recording(filename)
        print("Recording saved successfully!")


def human_controller(env: ServoEnv) -> None:
    """Human keyboard controller (A/D keys)"""
    env.start_recording()
    obs = env.reset()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
        
        # Get keyboard state
        keys = pygame.key.get_pressed()
        if keys[pygame.K_a]:  # Rotate left
            action = env.state.current_angle - 0.1
        elif keys[pygame.K_d]:  # Rotate right
            action = env.state.current_angle + 0.1
        else:
            action = env.state.current_angle
        
        # Take action
        obs, reward, done, _ = env.step(action)
        print(f"Reward: {reward:.3f}")
        
        time.sleep(0.01)
    
    env.stop_recording()
    env.save_recording('human_episode.hdf5')


def learned_controller(env: ServoEnv, policy_path: str = 'servo_policy.pth') -> None:
    """Controller that uses our trained policy"""
    # Load policy with CPU map location
    policy = SimplePolicy(image_size=240, use_qpos=True, qpos_dim=2).cpu()
    policy.load_state_dict(torch.load(policy_path, map_location=torch.device('cpu')))
    policy.eval()
    
    env.start_recording()
    obs = env.reset()
    running = True
    
    print("Running learned policy (press Q to quit)")
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
        
        # Get observation and convert to torch tensors
        image, qpos = obs
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        qpos_tensor = torch.from_numpy(qpos).float().unsqueeze(0)
        
        # Get action from policy
        with torch.no_grad():
            action = policy(image_tensor, qpos_tensor).item()
        
        # Take action
        obs, reward, done, _ = env.step(action)
        print(f"Reward: {reward:.3f}")
        
        pygame.event.pump()
        time.sleep(0.01)
    
    env.stop_recording()
    env.save_recording('learned_episode.hdf5')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Servo Flag Environment Controller')
    parser.add_argument('--controller', type=str, default='human',
                      choices=['human', 'scripted', 'learned'],
                      help='Type of controller to use (human, scripted, or learned)')
    parser.add_argument('--num_steps', type=int, default=10000,
                      help='Number of steps for scripted controller')
    parser.add_argument('--policy_path', type=str, default='servo_policy.pth',
                      help='Path to trained policy weights')
    parser.add_argument('--output', type=str, default=None,
                      help='Output filename for recording')
    
    args = parser.parse_args()
    
    env = ServoEnv()
    
    # Set output filename if not specified
    if args.output is None:
        args.output = f"{args.controller}_episode.hdf5"
    
    # Run selected controller
    if args.controller == "human":
        print("Human Controller:")
        print("  A/D - Rotate left/right")
        print("  Q   - Quit")
        human_controller(env)
    elif args.controller == "learned":
        learned_controller(env, args.policy_path)
    else:
        print(f"Running scripted controller for {args.num_steps} steps...")
        scripted_controller(env, num_steps=args.num_steps)
    
    pygame.quit()
