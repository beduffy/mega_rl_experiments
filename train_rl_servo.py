import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
import argparse
from simulated_pixel_servo_point_flag_at_target import ServoEnv

# Define experience tuple structure
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DQN(nn.Module):
    """DQN that takes in images and qpos and outputs Q-values for discretized actions"""
    def __init__(self, image_size=240, qpos_dim=2, num_actions=16):
        super().__init__()
        # Calculate conv output size
        def conv2d_output_size(size, kernel_size=3, stride=2):
            return ((size - kernel_size) // stride) + 1
        
        size1 = conv2d_output_size(image_size)  # 119
        size2 = conv2d_output_size(size1)       # 59
        flat_size = 32 * size2 * size2  # 32 channels * 59 * 59

        # CNN for processing images
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        # MLP for combining image features with qpos and outputting Q-values
        self.mlp = nn.Sequential(
            nn.Linear(flat_size + qpos_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)  # Output Q-value for each discrete action
        )
    
    def forward(self, image, qpos):
        x = self.conv(image)
        x = torch.cat([x, qpos], dim=1)
        return self.mlp(x)


class ReplayBuffer:
    """Experience replay buffer"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


def process_observation(obs):
    """Convert environment observation to torch tensors"""
    image, qpos = obs
    image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    qpos_tensor = torch.from_numpy(qpos).float().unsqueeze(0)
    return image_tensor, qpos_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_end', type=float, default=0.01)
    parser.add_argument('--epsilon_decay', type=float, default=0.995)
    parser.add_argument('--target_update', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # Initialize environment and networks
    env = ServoEnv()
    device = torch.device(args.device)
    
    # Create networks and optimizer
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
    
    # Create replay buffer
    replay_buffer = ReplayBuffer()
    
    # Training loop
    epsilon = args.epsilon_start
    
    for episode in range(args.num_episodes):
        obs = env.reset()
        episode_reward = 0
        
        while True:
            # Convert observation to torch tensors
            image_tensor, qpos_tensor = process_observation(obs)
            
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action_idx = random.randrange(16)
            else:
                with torch.no_grad():
                    q_values = policy_net(image_tensor.to(device), qpos_tensor.to(device))
                    action_idx = q_values.max(1)[1].item()
            
            # Convert discrete action to continuous
            action = (action_idx / 8.0 - 1.0) * np.pi
            
            # Take step in environment
            next_obs, reward, done, _ = env.step(action)
            
            # Store transition in replay buffer
            replay_buffer.push(obs, action_idx, reward, next_obs, done)
            
            # Train if enough samples
            if len(replay_buffer) >= args.batch_size:
                experiences = replay_buffer.sample(args.batch_size)
                
                # Process batch
                batch = Experience(*zip(*experiences))
                
                # Convert to tensors and train
                # ... (training step implementation)
            
            obs = next_obs
            episode_reward += reward
            
            if done:
                break
        
        # Update target network
        if episode % args.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Decay epsilon
        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)
        
        print(f"Episode {episode}: Reward = {episode_reward:.2f}, Epsilon = {epsilon:.2f}")


if __name__ == '__main__':
    main() 