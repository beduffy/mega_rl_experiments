import os
from datetime import datetime

import torch
import numpy as np
import h5py
import argparse
from torch import nn
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict


class SimplePolicy(nn.Module):
    """Neural network that maps camera observations to full 24-DoF joint angles.
    
    Input shapes:
        images: (batch_size, 3, 240, 240) - RGB images normalized to [0,1]
        qpos: (batch_size, 24) - Current joint positions (optional)
        
    Output shape:
        predictions: (batch_size, 24) - Predicted joint angles for all 24 DoF
    """
    def __init__(self, image_size=240, use_qpos=True, qpos_dim=24):
        super().__init__()
        # Calculate conv output size
        def conv2d_output_size(size, kernel_size=3, stride=2):
            return ((size - kernel_size) // stride) + 1
        
        size1 = conv2d_output_size(image_size)  # 119
        size2 = conv2d_output_size(size1)       # 59
        flat_size = 32 * size2 * size2  # 32 channels * 59 * 59

        # CNN processes images: (batch_size, 3, 240, 240) -> (batch_size, flat_size)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),  # (batch_size, 16, 119, 119)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2), # (batch_size, 32, 59, 59)
            nn.ReLU(),
            nn.Flatten(),  # (batch_size, 32 * 59 * 59)
        )

        # Expanded MLP for 24-DoF output
        combined_input_size = flat_size + qpos_dim
        self.mlp = nn.Sequential(
            nn.Linear(combined_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 24)  # Output all 24 joint angles
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
        return self.mlp(x)


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

class ServoDataset(Dataset):
    """Synthetic dataset for testing greet motion"""
    def __init__(self, num_samples=1000, image_size=240):
        # Generate random "images" (normalized to [0,1])
        self.images = torch.rand(num_samples, 3, image_size, image_size)
        # Use greet angles for all samples
        self.actions = GREET_ANGLES.repeat(num_samples, 1)
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], torch.zeros(24), self.actions[idx]  # Empty qpos


def train(policy, train_loader, num_epochs=50, lr=1e-4, device='cpu'):
    """Updated training function for 24-DoF control"""
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Get samples for visualization
    samples = []
    for batch in train_loader:
        if len(samples) < 10:
            img, qpos, target = batch
            samples.append((img[0], qpos[0], target[0]))
        else:
            break
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Store best loss and corresponding weights
    best_loss = float('inf')
    best_weights = None
    
    for epoch in range(num_epochs):
        total_loss = 0
        joint_errors = {name: 0.0 for name in JOINT_ORDER}
        
        for batch_idx, (images, qpos, targets) in enumerate(train_loader):
            images = images.to(device)
            qpos = qpos.to(device)
            targets = targets.to(device)
            
            outputs = policy(images, qpos)
            loss = criterion(outputs, targets)
            
            # Calculate per-joint errors
            with torch.no_grad():
                errors = torch.abs(outputs - targets).mean(dim=0)
                for i, name in enumerate(JOINT_ORDER):
                    joint_errors[name] += errors[i].item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print detailed joint errors
        avg_joint_errors = {k: v/len(train_loader) for k,v in joint_errors.items()}
        print(f'\nEpoch {epoch} Average Errors:')
        for joint, error in avg_joint_errors.items():
            print(f'  {joint:15}: {error:.4f}')
        
        # Update best weights
        epoch_loss = total_loss/len(train_loader)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_weights = policy.state_dict().copy()
        
        # Update sample predictions display
        with torch.no_grad():
            print(f'Epoch {epoch}: Loss: {total_loss/len(train_loader):.4f}')
            sample_pred = outputs[0].cpu().numpy()
            sample_target = targets[0].cpu().numpy()
            print(f"  Sample pred: {np.round(sample_pred, 3)}")
            print(f"  Target:      {np.round(sample_target, 3)}")
        
        # Show predictions on sample sequence
        with torch.no_grad():
            print(f'\nEpoch {epoch}:')
            print(f'  Average Loss: {total_loss/len(train_loader):.8f}')
            print("  Sample predictions vs targets (in radians):")
            
            # Get first sample predictions
            img, qpos, target = samples[0]
            img = img.unsqueeze(0).to(device)
            qpos = qpos.unsqueeze(0).to(device)
            pred = policy(img, qpos).squeeze().cpu().numpy()
            target_np = target.cpu().numpy()
            
            # Print first 5 joints for brevity
            for i in range(5):
                name = JOINT_ORDER[i]
                print(f"    {name:15}: {pred[i]:.3f} vs {target_np[i]:.3f}")
            print("    ... (remaining joints omitted for space) ...")
        
        # Save checkpoint every 100 epochs
        save_checkpoint_every_n_epochs = 10
        if (epoch + 1) % save_checkpoint_every_n_epochs == 0:
            checkpoint_path = f'checkpoints/servo_policy_{timestamp}_ep{epoch+1}.pth'
            torch.save(policy.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")


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
    dataset = ServoDataset(num_samples=1000)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Create and train policy
    policy = SimplePolicy(image_size=240, use_qpos=True, qpos_dim=24).to(device)
    train(policy, train_loader, num_epochs=args.num_epochs, lr=args.lr, device=device)
    
    # Save trained policy with timestamp and epochs
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'servo_policy_24dof_{timestamp}_ep{args.num_epochs}.pth'
    torch.save(policy.state_dict(), save_path)


if __name__ == '__main__':
    main()