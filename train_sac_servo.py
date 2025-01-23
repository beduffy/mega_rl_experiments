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
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        
        # Unpack states and next_states
        images, qpos = zip(*states)
        next_images, next_qpos = zip(*next_states)
        
        # Convert to tensors
        images = torch.FloatTensor(np.stack(images)).permute(0, 3, 1, 2) / 255.0
        qpos = torch.FloatTensor(np.stack(qpos))
        actions = torch.FloatTensor(np.stack(actions))
        rewards = torch.FloatTensor(np.stack(rewards))
        next_images = torch.FloatTensor(np.stack(next_images)).permute(0, 3, 1, 2) / 255.0
        next_qpos = torch.FloatTensor(np.stack(next_qpos))
        dones = torch.FloatTensor(np.stack(dones))
        
        return (images, qpos), actions, rewards, (next_images, next_qpos), dones
    
    def __len__(self):
        return len(self.buffer)

class SACPolicy(nn.Module):
    def __init__(self, image_size=240, qpos_dim=2, action_dim=1):
        super().__init__()
        def conv2d_size_out(size, kernel_size=3, stride=2):
            return ((size - kernel_size) // stride) + 1
        
        size1 = conv2d_size_out(image_size)  # 119
        size2 = conv2d_size_out(size1)       # 59
        self.flat_size = 32 * size2 * size2
        
        # Shared image encoder
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Actor network
        self.actor_net = nn.Sequential(
            nn.Linear(self.flat_size + qpos_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Linear(128, action_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0.0)
    
    def forward(self, image, qpos):
        features = self.conv(image)
        x = torch.cat([features, qpos], dim=1)
        x = self.actor_net(x)
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        
        return mean, log_std
    
    def sample(self, image, qpos):
        mean, log_std = self(image, qpos)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        
        # Constrain action to [-π, π]
        action = torch.tanh(x_t) * np.pi
        
        # Compute log probability, adding correction for tanh squashing
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        
        return action, log_prob

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
        
        # Initialize weights
        self.apply(self._init_weights)
    
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0.0)
    
    def forward(self, image, qpos, action):
        features = self.conv(image)
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
        image = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        qpos = torch.FloatTensor(qpos).unsqueeze(0).to(self.device)
        
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
        
        # Move to device
        image = image.to(self.device)
        qpos = qpos.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_image = next_image.to(self.device)
        next_qpos = next_qpos.to(self.device)
        done = done.to(self.device)
        
        # Update critics
        with torch.no_grad():
            next_action, next_log_prob = self.policy.sample(next_image, next_qpos)
            q1_next = self.critic1_target(next_image, next_qpos, next_action)
            q2_next = self.critic2_target(next_image, next_qpos, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            q_target = reward + (1 - done) * self.gamma * q_next
        
        # Critic 1 loss
        q1 = self.critic1(image, qpos, action)
        q1_loss = nn.MSELoss()(q1, q_target)
        
        # Critic 2 loss
        q2 = self.critic2(image, qpos, action)
        q2_loss = nn.MSELoss()(q2, q_target)
        
        # Update critics
        self.critic1_optim.zero_grad()
        q1_loss.backward()
        self.critic1_optim.step()
        
        self.critic2_optim.zero_grad()
        q2_loss.backward()
        self.critic2_optim.step()
        
        # Update policy
        new_action, log_prob = self.policy.sample(image, qpos)
        q1_new = self.critic1(image, qpos, new_action)
        q2_new = self.critic2(image, qpos, new_action)
        q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (self.alpha * log_prob - q_new).mean()
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
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
    
    agent = SAC(env, device, log_dir)
    
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