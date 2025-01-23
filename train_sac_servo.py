import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
import gymnasium as gym
from typing import Dict, Tuple
from collections import deque
import random
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

from simulated_pixel_servo_point_flag_at_target import ServoEnv

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        # Ensure consistent shapes when storing
        image, qpos = state
        next_image, next_qpos = next_state
        
        # Convert everything to numpy arrays with consistent shapes
        self.buffer.append((
            (np.array(image, dtype=np.float32),  # image shape: (H, W, C)
             np.array(qpos, dtype=np.float32).flatten()),  # qpos shape: (2,)
            np.array(action, dtype=np.float32).reshape(-1),  # action shape: (1,)
            np.array(reward, dtype=np.float32),  # reward shape: scalar
            (np.array(next_image, dtype=np.float32),  # next_image shape: (H, W, C)
             np.array(next_qpos, dtype=np.float32).flatten()),  # next_qpos shape: (2,)
            np.array(done, dtype=np.float32)  # done shape: scalar
        ))
    
    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        
        # Unpack and stack samples
        states, actions, rewards, next_states, dones = zip(*samples)
        
        # Unpack states and next_states
        images, qpos = zip(*states)
        next_images, next_qpos = zip(*next_states)
        
        # Convert to tensors with consistent shapes
        images = torch.FloatTensor(np.stack(images)) / 255.0  # (B, H, W, C)
        images = images.permute(0, 3, 1, 2)  # Convert to (B, C, H, W)
        qpos = torch.FloatTensor(np.stack(qpos))  # (B, 2)
        
        next_images = torch.FloatTensor(np.stack(next_images)) / 255.0  # (B, H, W, C)
        next_images = next_images.permute(0, 3, 1, 2)  # Convert to (B, C, H, W)
        next_qpos = torch.FloatTensor(np.stack(next_qpos))  # (B, 2)
        
        actions = torch.FloatTensor(np.stack(actions)).unsqueeze(-1)  # (B, 1)
        rewards = torch.FloatTensor(np.stack(rewards))  # (B,)
        dones = torch.FloatTensor(np.stack(dones))  # (B,)
        
        return (images, qpos), actions, rewards, (next_images, next_qpos), dones
    
    def __len__(self):
        return len(self.buffer)

class SACPolicy(nn.Module):
    def __init__(self, image_size=240, qpos_dim=2, action_dim=1):
        super().__init__()
        def conv2d_size_out(size, kernel_size=3, stride=2):
            return ((size - kernel_size) // stride) + 1
        
        size1 = conv2d_size_out(image_size)
        size2 = conv2d_size_out(size1)
        self.flat_size = 32 * size2 * size2
        
        # More stable CNN with proper normalization
        self.conv = nn.Sequential(
            # First conv block with smaller initial features
            nn.Conv2d(3, 8, kernel_size=3, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Add pooling to reduce feature magnitude
            
            # Second conv block
            nn.Conv2d(8, 16, kernel_size=3, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Final conv block
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Force output size
            nn.Flatten()
        )
        
        # Recalculate flat size after adaptive pooling
        self.flat_size = 32 * 4 * 4
        
        # Actor network with proper normalization
        self.actor_net = nn.Sequential(
            nn.Linear(self.flat_size + qpos_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),  # Add dropout for regularization
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Output layers with careful initialization
        self.mean = nn.Linear(128, action_dim)
        torch.nn.init.uniform_(self.mean.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.mean.bias, -1e-3, 1e-3)
        
        self.log_std = nn.Linear(128, action_dim)
        torch.nn.init.uniform_(self.log_std.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_std.bias, -1e-3, 1e-3)
    
    def forward(self, image, qpos):
        # Ensure inputs are in correct range
        image = image.clamp(0, 1)  # Normalize image to [0,1]
        qpos = qpos.clamp(-10, 10)  # Reasonable range for angles
        
        features = self.conv(image)
        if torch.isnan(features).any():
            print("NaN detected in features!")
            return None, None
        
        if len(qpos.shape) > 2:
            qpos = qpos.reshape(qpos.shape[0], -1)
        
        x = torch.cat([features, qpos], dim=1)
        x = self.actor_net(x)
        
        # Bound outputs for stability
        mean = torch.tanh(self.mean(x))
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -5, 2)  # Less aggressive clamping
        
        return mean, log_std
    
    def sample(self, image, qpos):
        mean, log_std = self(image, qpos)
        
        if mean is None or log_std is None:  # Check for NaN detection
            raise ValueError("NaN detected in network output")
        
        # More stable std computation
        std = torch.exp(log_std).clamp(1e-4, 2.0)
        
        # Create distribution and sample
        try:
            normal = Normal(mean, std)
            x_t = normal.rsample()
            
            # Bound actions to [-π, π]
            action = torch.tanh(x_t) * np.pi
            
            # Compute log prob with better numerical stability
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(1 - action.pow(2).clamp(max=0.9999) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
            
            return action, log_prob
            
        except Exception as e:
            print(f"\nError in sampling:")
            print(f"mean shape: {mean.shape}, values: {mean}")
            print(f"std shape: {std.shape}, values: {std}")
            raise e

class Critic(nn.Module):
    def __init__(self, image_size=240, qpos_dim=2, action_dim=1):
        super().__init__()
        def conv2d_size_out(size, kernel_size=3, stride=2):
            return ((size - kernel_size) // stride) + 1
        
        size1 = conv2d_size_out(image_size)
        size2 = conv2d_size_out(size1)
        self.flat_size = 32 * size2 * size2
        
        # Shared image encoder
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Q network
        self.q_net = nn.Sequential(
            nn.Linear(self.flat_size + qpos_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, image, qpos, action):
        # Print shapes for debugging
        # print(f"Image shape: {image.shape}")
        # print(f"Qpos shape: {qpos.shape}")
        # print(f"Action shape: {action.shape}")
        
        features = self.conv(image)  # [B, flat_size]
        
        # Ensure all inputs are 2D: [batch_size, features]
        if len(qpos.shape) > 2:
            qpos = qpos.reshape(qpos.shape[0], -1)
        if len(action.shape) > 2:
            action = action.reshape(action.shape[0], -1)
        
        # print(f"Features shape: {features.shape}")
        # print(f"Reshaped qpos shape: {qpos.shape}")
        # print(f"Reshaped action shape: {action.shape}")
        
        x = torch.cat([features, qpos, action], dim=1)
        return self.q_net(x)

class SAC:
    def __init__(
        self,
        env,
        device,
        log_dir,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        buffer_size: int = 100000,
        batch_size: int = 256,
    ):
        self.env = env
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        
        # Initialize networks
        self.policy = SACPolicy().to(device)
        self.critic1 = Critic().to(device)
        self.critic2 = Critic().to(device)
        self.critic1_target = Critic().to(device)
        self.critic2_target = Critic().to(device)
        
        # Copy parameters to target networks
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Initialize optimizers
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=lr)
        self.critic1_optim = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optim = optim.Adam(self.critic2.parameters(), lr=lr)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Initialize tensorboard
        self.writer = SummaryWriter(log_dir)
        self.train_steps = 0
    
    def select_action(self, state, evaluate=False):
        image, qpos = state
        # print("\nDEBUG select_action:")
        # print(f"Original image shape: {image.shape}")
        # print(f"Original qpos shape: {qpos.shape}")
        
        # Add batch dimension and normalize
        image = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        qpos = torch.FloatTensor(qpos).flatten().unsqueeze(0).to(self.device)
        
        # print(f"Processed image shape: {image.shape}")
        # print(f"Processed qpos shape: {qpos.shape}")
        
        if evaluate:
            with torch.no_grad():
                mean, _ = self.policy(image, qpos)
                return torch.tanh(mean) * np.pi
        else:
            with torch.no_grad():
                action, _ = self.policy.sample(image, qpos)
                return action
    
    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample from replay buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        (image, qpos) = state
        (next_image, next_qpos) = next_state
        
        # Move to device and ensure shapes
        image = image.to(self.device)  # [B, C, H, W]
        qpos = qpos.to(self.device)    # [B, 2]
        action = action.to(self.device) # [B, 1]
        reward = reward.unsqueeze(-1).to(self.device)  # [B, 1]
        next_image = next_image.to(self.device)
        next_qpos = next_qpos.to(self.device)
        done = done.unsqueeze(-1).to(self.device)  # [B, 1]
        
        # Update critics
        with torch.no_grad():
            next_action, next_log_prob = self.policy.sample(next_image, next_qpos)
            q1_next = self.critic1_target(next_image, next_qpos, next_action)
            q2_next = self.critic2_target(next_image, next_qpos, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            q_target = reward + (1 - done) * self.gamma * q_next
        
        # Critic 1 loss
        q1 = self.critic1(image, qpos, action)
        q1_loss = nn.MSELoss()(q1, q_target.detach())
        
        # Critic 2 loss
        q2 = self.critic2(image, qpos, action)
        q2_loss = nn.MSELoss()(q2, q_target.detach())
        
        # Update critics
        self.critic1_optim.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)  # Add gradient clipping
        self.critic1_optim.step()
        
        self.critic2_optim.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)  # Add gradient clipping
        self.critic2_optim.step()
        
        # Update policy
        new_action, log_prob = self.policy.sample(image, qpos)
        q1_new = self.critic1(image, qpos, new_action)
        q2_new = self.critic2(image, qpos, new_action)
        q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (self.alpha * log_prob - q_new).mean()
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)  # Add gradient clipping
        self.policy_optim.step()
        
        # Update target networks
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Log metrics
        self.writer.add_scalar('Loss/critic1', q1_loss.item(), self.train_steps)
        self.writer.add_scalar('Loss/critic2', q2_loss.item(), self.train_steps)
        self.writer.add_scalar('Loss/policy', policy_loss.item(), self.train_steps)
        
        self.train_steps += 1

def main():
    # Create directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('runs', f'sac_servo_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize environment and agent
    env = ServoEnv(render_mode=None)  # Headless mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    agent = SAC(
        env=env,
        device=device,
        log_dir=log_dir,
        lr=1e-4,  # Slightly higher learning rate
        gamma=0.99,
        tau=0.005,
        alpha=0.2,  # Standard temperature
        buffer_size=100000,
        batch_size=32  # Smaller batch size for stability
    )
    
    # Training loop
    episodes = 1000
    eval_interval = 10
    max_steps = 200
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
            
            # Store transition
            agent.replay_buffer.push(state, action.cpu().numpy(), reward, next_state, done)
            
            # Train agent
            agent.train_step()
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Log episode metrics
        agent.writer.add_scalar('Reward/train', episode_reward, episode)
        
        # Evaluate
        if episode % eval_interval == 0:
            eval_rewards = []
            eval_env = ServoEnv(render_mode='human')  # Render during eval
            
            for _ in range(5):
                state = eval_env.reset()
                eval_reward = 0
                
                while True:
                    action = agent.select_action(state, evaluate=True)
                    next_state, reward, done, _ = eval_env.step(action.cpu().numpy()[0])
                    eval_reward += reward
                    state = next_state
                    if done:
                        break
                
                eval_rewards.append(eval_reward)
            
            mean_reward = np.mean(eval_rewards)
            agent.writer.add_scalar('Reward/eval', mean_reward, episode)
            print(f"Episode {episode}: Eval reward = {mean_reward:.2f}")

if __name__ == '__main__':
    main() 