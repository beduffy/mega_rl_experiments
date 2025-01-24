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
import argparse
import time
import torch.nn.functional as F

from simulated_pixel_servo_point_flag_at_target import ServoEnv

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
        self.batch_images = None
        self.batch_next_images = None
    
    def push(self, state, action, reward, next_state, done):
        # Ensure images are in CHW format when storing
        image, qpos = state
        next_image, next_qpos = next_state
        
        # Convert HWC to CHW format
        image = np.transpose(image, (2, 0, 1))
        next_image = np.transpose(next_image, (2, 0, 1))
        
        self.buffer.append((
            (image.copy(), np.array(qpos, dtype=np.float32).flatten()),
            np.array(action, dtype=np.float32).reshape(-1),
            np.array(reward, dtype=np.float32),
            (next_image.copy(), np.array(next_qpos, dtype=np.float32).flatten()),
            np.array(done, dtype=np.float32)
        ))
    
    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        
        # Initialize arrays if not done yet
        if self.batch_images is None:
            # Get shapes from first sample (now in CHW format)
            img_shape = samples[0][0][0].shape
            self.batch_images = np.zeros((batch_size,) + img_shape, dtype=np.float32)
            self.batch_next_images = np.zeros((batch_size,) + img_shape, dtype=np.float32)
        
        # Batch processing (images already in CHW format)
        for i, (state, action, reward, next_state, done) in enumerate(samples):
            self.batch_images[i] = state[0]
            self.batch_next_images[i] = next_state[0]
        
        qpos = np.stack([s[0][1] for s in samples])
        next_qpos = np.stack([s[3][1] for s in samples])
        actions = np.stack([s[1] for s in samples])
        rewards = np.stack([s[2] for s in samples])
        dones = np.stack([s[4] for s in samples])
        
        # Convert to tensors (images already in CHW format)
        return (
            (torch.from_numpy(self.batch_images).to(torch.float32) / 255.0,
             torch.from_numpy(qpos).to(torch.float32)),
            torch.from_numpy(actions).to(torch.float32),
            torch.from_numpy(rewards).to(torch.float32),
            (torch.from_numpy(self.batch_next_images).to(torch.float32) / 255.0,
             torch.from_numpy(next_qpos).to(torch.float32)),
            torch.from_numpy(dones).to(torch.float32)
        )
    
    def __len__(self):
        return len(self.buffer)

class SACPolicy(nn.Module):
    def __init__(self, image_size=240, qpos_dim=2, action_dim=1):
        super().__init__()
        
        # Much smaller CNN (same as critic for simplicity)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten()
        )
        
        self.flat_size = 32 * 3 * 3  # 288
        
        # Smaller actor network
        self.actor_net = nn.Sequential(
            nn.Linear(self.flat_size + qpos_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Output layers
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Linear(128, action_dim)
        
        # Initialize output layers
        torch.nn.init.uniform_(self.mean.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.mean.bias, -1e-3, 1e-3)
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
        
        # Much smaller CNN
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=8, stride=4),  # Bigger stride to reduce dimensions faster
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),  # Force small output size
            nn.Flatten()
        )
        
        # Calculate flat size
        self.flat_size = 32 * 3 * 3  # Much smaller: 288
        
        # Smaller Q network
        self.q_net = nn.Sequential(
            nn.Linear(self.flat_size + qpos_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, image, qpos, action):
        features = self.conv(image)
        if len(qpos.shape) > 2:
            qpos = qpos.reshape(qpos.shape[0], -1)
        if len(action.shape) > 2:
            action = action.reshape(action.shape[0], -1)
        
        x = torch.cat([features, qpos, action], dim=1)
        return self.q_net(x)

class SAC:
    def __init__(self, env, device, log_dir, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2,
                 buffer_size=100000, batch_size=64):
        self.env = env
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        
        # Move models to device immediately after creation
        self.policy = SACPolicy().to(device)
        self.critic1 = Critic().to(device)
        self.critic2 = Critic().to(device)
        self.critic1_target = Critic().to(device)
        self.critic2_target = Critic().to(device)
        
        # Hard copy parameters
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Create optimizers after moving models to device
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.writer = SummaryWriter(log_dir)
        self.train_steps = 0
    
    def select_action(self, state, evaluate=False):
        image, qpos = state
        
        # Convert HWC to CHW format and move to device
        image = torch.FloatTensor(np.transpose(image, (2, 0, 1))).unsqueeze(0).to(self.device) / 255.0
        qpos = torch.FloatTensor(qpos).flatten().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if evaluate:
                action, _ = self.policy.sample(image, qpos)
            else:
                action, _ = self.policy.sample(image, qpos)
        return action
    
    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Get batch and move to device
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        (image, qpos) = state
        (next_image, next_qpos) = next_state
        
        # Move everything to device
        image = image.to(self.device)
        qpos = qpos.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_image = next_image.to(self.device)
        next_qpos = next_qpos.to(self.device)
        done = done.to(self.device)
        
        # Pre-compute policy outputs for both current and next state
        with torch.no_grad():
            next_action, next_log_prob = self.policy.sample(next_image, next_qpos)
            current_action, current_log_prob = self.policy.sample(image, qpos)
        
        # Update critics (in parallel)
        with torch.no_grad():
            q1_next = self.critic1_target(next_image, next_qpos, next_action)
            q2_next = self.critic2_target(next_image, next_qpos, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            q_target = reward.unsqueeze(-1) + (1 - done.unsqueeze(-1)) * self.gamma * q_next
        
        # Critic updates (combined for efficiency)
        q1 = self.critic1(image, qpos, action)
        q2 = self.critic2(image, qpos, action)
        q1_loss = F.mse_loss(q1, q_target)
        q2_loss = F.mse_loss(q2, q_target)
        
        # Combined backward pass for critics
        self.critic1_optimizer.zero_grad(set_to_none=True)
        self.critic2_optimizer.zero_grad(set_to_none=True)
        q1_loss.backward()
        q2_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()
        
        # Policy update (using pre-computed actions)
        q1_pi = self.critic1(image, qpos, current_action)
        q2_pi = self.critic2(image, qpos, current_action)
        min_q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (self.alpha * current_log_prob - min_q_pi).mean()
        
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Soft update target networks (less frequently)
        if self.train_steps % 2 == 0:  # Update every other step
            with torch.no_grad():
                for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        self.train_steps += 1

def parse_args():
    parser = argparse.ArgumentParser(description='Train SAC on Servo Environment')
    
    # Environment parameters
    parser.add_argument('--render-mode', type=str, default=None, choices=[None, 'human'],
                       help='Render mode for environment. None for headless, human for rendering')
    parser.add_argument('--eval-render-mode', type=str, default='human', choices=[None, 'human'],
                       help='Render mode for evaluation episodes')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of episodes to train')
    parser.add_argument('--eval-interval', type=int, default=10,
                       help='Evaluate every N episodes')
    parser.add_argument('--eval-episodes', type=int, default=5,
                       help='Number of evaluation episodes')
    parser.add_argument('--max-steps', type=int, default=200,
                       help='Maximum steps per episode')
    
    # SAC hyperparameters
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--alpha', type=float, default=0.2,
                       help='Temperature parameter')
    parser.add_argument('--tau', type=float, default=0.005,
                       help='Target network update rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--buffer-size', type=int, default=100000,
                       help='Replay buffer size')
    
    # Device and logging
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--log-dir', type=str, default='runs',
                       help='Directory for tensorboard logs')
    
    return parser.parse_args()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"\nUsing device: {device}")
    
    # Create environment
    env = ServoEnv(render_mode=args.render_mode)
    
    # Create directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.log_dir, f'sac_servo_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f'{arg}: {value}\n')
    
    # Initialize agent
    agent = SAC(
        env=env,
        device=device,
        log_dir=log_dir,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size
    )
    
    # Print model device info
    print("\nModel devices:")
    print(f"Policy: {next(agent.policy.parameters()).device}")
    print(f"Critic1: {next(agent.critic1.parameters()).device}")
    print(f"Critic2: {next(agent.critic2.parameters()).device}")
    
    # Add detailed timing
    timings = {
        'env_step': 0,
        'select_action': 0,
        'buffer_push': 0,
        'train_step': 0
    }
    counts = {k: 0 for k in timings.keys()}
    
    start_time = time.time()
    total_steps = 0
    last_print_time = start_time
    last_print_steps = 0
    
    print(f"\nStarting training with device: {args.device}")
    print(f"Render mode: {args.render_mode}")
    # Print model sizes
    print("\nModel sizes (parameters):")
    print(f"Policy CNN: {count_parameters(agent.policy.conv):,}")
    print(f"Policy Actor: {count_parameters(agent.policy.actor_net):,}")
    print(f"Critic: {count_parameters(agent.critic1):,}")
    print(f"Total: {count_parameters(agent.policy) + count_parameters(agent.critic1) * 2:,}")
    
    # Training loop
    for episode in range(args.episodes):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        for step in range(args.max_steps):
            total_steps += 1
            episode_steps += 1
            
            # Time select_action
            t0 = time.time()
            action = agent.select_action(state)
            timings['select_action'] += time.time() - t0
            counts['select_action'] += 1
            
            # Time env_step
            t0 = time.time()
            next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
            timings['env_step'] += time.time() - t0
            counts['env_step'] += 1
            
            # Time buffer_push
            t0 = time.time()
            agent.replay_buffer.push(state, action.cpu().numpy(), reward, next_state, done)
            timings['buffer_push'] += time.time() - t0
            counts['buffer_push'] += 1
            
            # Time train_step
            t0 = time.time()
            agent.train_step()
            timings['train_step'] += time.time() - t0
            counts['train_step'] += 1
            
            episode_reward += reward
            state = next_state
            
            # Print progress and timing stats every 20 seconds
            current_time = time.time()
            if current_time - last_print_time >= 20:
                steps_since_last_print = total_steps - last_print_steps
                time_elapsed = current_time - last_print_time
                fps = steps_since_last_print / time_elapsed
                
                print(f"\nProgress Update:")
                print(f"Episode: {episode}/{args.episodes}")
                print(f"Total Steps: {total_steps}")
                print(f"FPS: {fps:.1f}")
                print(f"Buffer Size: {len(agent.replay_buffer)}/{args.buffer_size}")
                print(f"Current Episode Steps: {episode_steps}")
                print(f"Current Episode Reward: {episode_reward:.2f}")
                
                # Print timing breakdown
                print("\nTiming Breakdown (ms per call):")
                for k in timings:
                    if counts[k] > 0:
                        avg_time = (timings[k] / counts[k]) * 1000
                        print(f"{k}: {avg_time:.2f}ms")
                
                # Reset timing stats
                timings = {k: 0 for k in timings}
                counts = {k: 0 for k in counts}
                
                last_print_time = current_time
                last_print_steps = total_steps
            
            if done:
                break
        
        # Log episode metrics
        agent.writer.add_scalar('Reward/train', episode_reward, episode)
        agent.writer.add_scalar('Steps/train', episode_steps, episode)
        
        # Evaluate
        if episode % args.eval_interval == 0:
            eval_rewards = []
            eval_env = ServoEnv(render_mode=args.eval_render_mode)
            
            for _ in range(args.eval_episodes):
                eval_state = eval_env.reset()
                eval_reward = 0
                
                while True:
                    eval_action = agent.select_action(eval_state, evaluate=True)
                    eval_next_state, eval_r, eval_done, _ = eval_env.step(eval_action.cpu().numpy()[0])
                    eval_reward += eval_r
                    eval_state = eval_next_state
                    if eval_done:
                        break
                
                eval_rewards.append(eval_reward)
            
            mean_reward = np.mean(eval_rewards)
            agent.writer.add_scalar('Reward/eval', mean_reward, episode)
            print(f"\nEpisode {episode}: Eval reward = {mean_reward:.2f}")


if __name__ == '__main__':
    main() 