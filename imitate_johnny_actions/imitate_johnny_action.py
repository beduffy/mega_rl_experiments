import os
from datetime import datetime

import torch
import numpy as np
import h5py
import argparse
from torch import nn
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict


class SequencePolicy(nn.Module):
    """Predicts N future action steps from current observation"""
    def __init__(self, image_size=240, use_qpos=True, qpos_dim=24, pred_steps=3):
        super().__init__()
        # Keep existing CNN backbone
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Expanded MLP for multi-step prediction
        self.pred_steps = pred_steps
        combined_dim = 32*59*59 + qpos_dim
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 24 * pred_steps)  # Output 24 * pred_steps joint angles
        )
        
        self.use_qpos = use_qpos
        self.joint_order = [
            'r_hip_yaw', 'r_hip_roll', 'r_hip_pitch', 'r_knee', 'r_ank_pitch', 'r_ank_roll',
            'l_hip_yaw', 'l_hip_roll', 'l_hip_pitch', 'l_knee', 'l_ank_pitch', 'l_ank_roll',
            'head_pan', 'head_tilt',
            'r_sho_pitch', 'r_sho_roll', 'r_el_pitch', 'r_el_yaw', 'r_gripper',
            'l_sho_pitch', 'l_sho_roll', 'l_el_pitch', 'l_el_yaw', 'l_gripper'
        ]
    
    def forward(self, images, qpos=None):
        x = self.conv(images)
        if self.use_qpos and qpos is not None:
            x = torch.cat([x, qpos], dim=1)
        # Reshape output to (batch_size, pred_steps, 24)
        return self.mlp(x).view(-1, self.pred_steps, 24)


# Define joint order matching the URDF structure
JOINT_ORDER = [
    'r_hip_yaw', 'r_hip_roll', 'r_hip_pitch', 'r_knee', 'r_ank_pitch', 'r_ank_roll',
    'l_hip_yaw', 'l_hip_roll', 'l_hip_pitch', 'l_knee', 'l_ank_pitch', 'l_ank_roll',
    'head_pan', 'head_tilt',
    'r_sho_pitch', 'r_sho_roll', 'r_el_pitch', 'r_el_yaw', 'r_gripper',
    'l_sho_pitch', 'l_sho_roll', 'l_el_pitch', 'l_el_yaw', 'l_gripper'
]

# Hardcoded demonstration data
GREET_ANGLES = torch.tensor([
    0.0, 0.0, 0.7120943226666667, -1.0890854346666667, -0.5864306186666667, 0.0,  # Right leg
    0.0, 0.0, -0.7120943226666667, 1.0890854346666667, 0.5864306186666667, 0.0,   # Left leg
    0.0, 0.0,  # Head
    0.16755160533333335, 1.3823007440000001, 0.0, 1.25663704, 0.0,  # Right arm
    -1.9896753133333334, 0.0, 0.0, -1.8011797573333335, 0.0  # Left arm
], dtype=torch.float32)

all_greet_action_lines = [
    {'l_ank_roll': 0.0, 'r_ank_roll': 0.0, 'l_ank_pitch': 0.5864306186666667, 'r_ank_pitch': -0.5864306186666667, 'l_knee': 1.0890854346666667, 'r_knee': -1.0890854346666667, 'l_hip_pitch': -0.7120943226666667, 'r_hip_pitch': 0.7120943226666667, 'l_hip_roll': 0.0, 'r_hip_roll': 0.0, 'l_hip_yaw': 0.0, 'r_hip_yaw': 0.0, 'l_sho_pitch': 0.16755160533333335, 'r_sho_pitch': -1.9896753133333334, 'l_sho_roll': 1.3823007440000001, 'r_sho_roll': 0.0, 'l_el_pitch': 0.0, 'r_el_pitch': 0.0, 'l_el_yaw': -1.8011797573333335, 'r_el_yaw': 1.25663704, 'l_gripper': 0.0, 'r_gripper': 0.0},
    {'l_ank_roll': 0.0, 'r_ank_roll': 0.0, 'l_ank_pitch': 0.5864306186666667, 'r_ank_pitch': -0.5864306186666667, 'l_knee': 1.0890854346666667, 'r_knee': -1.0890854346666667, 'l_hip_pitch': -0.7120943226666667, 'r_hip_pitch': 0.7120943226666667, 'l_hip_roll': 0.0, 'r_hip_roll': 0.0, 'l_hip_yaw': 0.0, 'r_hip_yaw': 0.0, 'l_sho_pitch': 0.16755160533333335, 'r_sho_pitch': -1.9896753133333334, 'l_sho_roll': 1.3823007440000001, 'r_sho_roll': 0.0, 'l_el_pitch': 0.0, 'r_el_pitch': 0.0, 'l_el_yaw': -1.8011797573333335, 'r_el_yaw': 1.25663704, 'l_gripper': 0.0, 'r_gripper': 0.0},
    {'l_ank_roll': 0.0, 'r_ank_roll': 0.0, 'l_ank_pitch': 0.5864306186666667, 'r_ank_pitch': -0.5864306186666667, 'l_knee': 1.0890854346666667, 'r_knee': -1.0890854346666667, 'l_hip_pitch': -0.7120943226666667, 'r_hip_pitch': 0.7120943226666667, 'l_hip_roll': 0.0, 'r_hip_roll': 0.0, 'l_hip_yaw': 0.0, 'r_hip_yaw': 0.0, 'l_sho_pitch': 0.16755160533333335, 'r_sho_pitch': -1.9896753133333334, 'l_sho_roll': 1.3823007440000001, 'r_sho_roll': 0.0, 'l_el_pitch': 0.0, 'r_el_pitch': 0.0, 'l_el_yaw': -1.8011797573333335, 'r_el_yaw': 1.88495556, 'l_gripper': 0.0, 'r_gripper': 0.0},
    {'l_ank_roll': 0.0, 'r_ank_roll': 0.0, 'l_ank_pitch': 0.5864306186666667, 'r_ank_pitch': -0.5864306186666667, 'l_knee': 1.0890854346666667, 'r_knee': -1.0890854346666667, 'l_hip_pitch': -0.7120943226666667, 'r_hip_pitch': 0.7120943226666667, 'l_hip_roll': 0.0, 'r_hip_roll': 0.0, 'l_hip_yaw': 0.0, 'r_hip_yaw': 0.0, 'l_sho_pitch': 0.16755160533333335, 'r_sho_pitch': -1.9896753133333334, 'l_sho_roll': 1.3823007440000001, 'r_sho_roll': 0.0, 'l_el_pitch': 0.0, 'r_el_pitch': 0.0, 'l_el_yaw': -1.8011797573333335, 'r_el_yaw': 1.25663704, 'l_gripper': 0.0, 'r_gripper': 0.0},
    {'l_ank_roll': 0.0, 'r_ank_roll': 0.0, 'l_ank_pitch': 0.5864306186666667, 'r_ank_pitch': -0.5864306186666667, 'l_knee': 1.0890854346666667, 'r_knee': -1.0890854346666667, 'l_hip_pitch': -0.7120943226666667, 'r_hip_pitch': 0.7120943226666667, 'l_hip_roll': 0.0, 'r_hip_roll': 0.0, 'l_hip_yaw': 0.0, 'r_hip_yaw': 0.0, 'l_sho_pitch': 0.16755160533333335, 'r_sho_pitch': -1.9896753133333334, 'l_sho_roll': 1.3823007440000001, 'r_sho_roll': 0.0, 'l_el_pitch': 0.0, 'r_el_pitch': 0.0, 'l_el_yaw': -1.8011797573333335, 'r_el_yaw': 1.88495556, 'l_gripper': 0.0, 'r_gripper': 0.0},
    {'l_ank_roll': 0.0, 'r_ank_roll': 0.0, 'l_ank_pitch': 0.5864306186666667, 'r_ank_pitch': -0.5864306186666667, 'l_knee': 1.0890854346666667, 'r_knee': -1.0890854346666667, 'l_hip_pitch': -0.7120943226666667, 'r_hip_pitch': 0.7120943226666667, 'l_hip_roll': 0.0, 'r_hip_roll': 0.0, 'l_hip_yaw': 0.0, 'r_hip_yaw': 0.0, 'l_sho_pitch': 0.16755160533333335, 'r_sho_pitch': -1.9896753133333334, 'l_sho_roll': 1.3823007440000001, 'r_sho_roll': 0.0, 'l_el_pitch': 0.0, 'r_el_pitch': 0.0, 'l_el_yaw': -1.8011797573333335, 'r_el_yaw': 1.25663704, 'l_gripper': 0.0, 'r_gripper': 0.0},
    {'l_ank_roll': 0.0, 'r_ank_roll': 0.0, 'l_ank_pitch': 0.5864306186666667, 'r_ank_pitch': -0.5864306186666667, 'l_knee': 1.0890854346666667, 'r_knee': -1.0890854346666667, 'l_hip_pitch': -0.7120943226666667, 'r_hip_pitch': 0.7120943226666667, 'l_hip_roll': 0.0, 'r_hip_roll': 0.0, 'l_hip_yaw': 0.0, 'r_hip_yaw': 0.0, 'l_sho_pitch': 0.16755160533333335, 'r_sho_pitch': -1.9896753133333334, 'l_sho_roll': 1.3823007440000001, 'r_sho_roll': 0.0, 'l_el_pitch': 0.0, 'r_el_pitch': 0.0, 'l_el_yaw': -1.8011797573333335, 'r_el_yaw': 1.25663704, 'l_gripper': 0.0, 'r_gripper': 0.0},
    {'l_ank_roll': 0.0, 'r_ank_roll': 0.0, 'l_ank_pitch': 0.5864306186666667, 'r_ank_pitch': -0.5864306186666667, 'l_knee': 1.0890854346666667, 'r_knee': -1.0890854346666667, 'l_hip_pitch': -0.7120943226666667, 'r_hip_pitch': 0.7120943226666667, 'l_hip_roll': 0.0, 'r_hip_roll': 0.0, 'l_hip_yaw': 0.0, 'r_hip_yaw': 0.0, 'l_sho_pitch': 0.16755160533333335, 'r_sho_pitch': -0.16755160533333335, 'l_sho_roll': 1.3823007440000001, 'r_sho_roll': -1.3823007440000001, 'l_el_pitch': 0.0, 'r_el_pitch': 0.0, 'l_el_yaw': -1.8011797573333335, 'r_el_yaw': 1.8011797573333335, 'l_gripper': 0.0, 'r_gripper': 0.0}
]

# Modified dataset class for sequence prediction
class ServoDataset(Dataset):
    """Dataset for sequence prediction with sliding window"""
    def __init__(self, action_sequences, num_samples=1000, window_size=3, image_size=240):
        # Convert each sequence from list of dicts to list of tensors
        self.action_sequences = [
            [self.dict_to_tensor(step) for step in seq] 
            for seq in action_sequences
        ]
        self.window_size = window_size
        self.images = torch.rand(num_samples, 3, image_size, image_size)
        
        # Create sequence targets
        self.targets = []
        for _ in range(num_samples):
            seq_idx = np.random.randint(0, len(action_sequences))
            start = np.random.randint(0, len(action_sequences[seq_idx]) - window_size)
            self.targets.append(torch.stack(self.action_sequences[seq_idx][start:start+window_size]))
        
    def dict_to_tensor(self, action_dict):
        return torch.tensor([
            action_dict.get(name, 0.0)  # Use 0.0 as default for missing joints
            for name in JOINT_ORDER
        ], dtype=torch.float32)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], torch.zeros(24), self.targets[idx]  # (image, qpos, sequence_target)


def train(policy, train_loader, num_epochs=50, lr=1e-4, device='cpu'):
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_loss = float('inf')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for epoch in range(num_epochs):
        policy.train()
        total_loss = 0
        joint_errors = {name: 0.0 for name in JOINT_ORDER}
        
        for batch_idx, (images, qpos, targets) in enumerate(train_loader):
            images = images.to(device)
            qpos = qpos.to(device)
            targets = targets.to(device)
            
            # Predict sequence: (B, T, 24)
            preds = policy(images, qpos)
            
            # Reshape targets if needed and calculate loss
            loss = criterion(preds, targets)  # Now both are (B, T, 24)
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate per-joint errors
            with torch.no_grad():
                errors = torch.abs(preds - targets).mean(dim=(0,1))  # Average over batch and time
                for i, name in enumerate(JOINT_ORDER):
                    joint_errors[name] += errors[i].item()
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            ckpt_path = f'checkpoints/policy_epoch{epoch}_{timestamp}.pth'
            torch.save(policy.state_dict(), ckpt_path)
        
        # Print diagnostics
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch}: Loss: {avg_loss:.14f}')
        
        # Print sample predictions
        with torch.no_grad():
            sample_img, sample_qpos, sample_target = next(iter(train_loader))
            sample_pred = policy(sample_img[:1].to(device), sample_qpos[:1].to(device))
            
            print("\nSample prediction vs target (first and last timesteps):")
            for j in range(5):  # First 5 joints
                name = JOINT_ORDER[j]
                pred_first = sample_pred[0,0,j].item()
                target_first = sample_target[0,0,j].item()
                pred_last = sample_pred[0,-1,j].item()
                target_last = sample_target[0,-1,j].item()
                print(f"  {name:15}: First: {pred_first:.5f} vs {target_first:.5f}, Last: {pred_last:.5f} vs {target_last:.5f}")

        # Update best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(policy.state_dict(), f'best_policy_{timestamp}.pth')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='',  # Keep argument but don't use it
                      help='Not used, preserved for compatibility')
    parser.add_argument('--num_epochs', type=int, default=500,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                      help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu',
                      choices=['cpu', 'cuda'], 
                      help='Device to train on (cpu or cuda)')
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Training on {device}")
    
    # Create synthetic dataset
    greet_sequences = [all_greet_action_lines]  # Can add more sequences
    dataset = ServoDataset(action_sequences=greet_sequences, num_samples=1000)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Create and train policy
    policy = SequencePolicy(image_size=240, use_qpos=True, qpos_dim=24, pred_steps=3).to(device)
    train(policy, train_loader, num_epochs=args.num_epochs, lr=args.lr, device=device)
    
    # Save trained policy with timestamp and epochs
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'servo_policy_24dof_{timestamp}_ep{args.num_epochs}.pth'
    torch.save(policy.state_dict(), save_path)


if __name__ == '__main__':
    main()