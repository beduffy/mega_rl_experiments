import os
import sys
import time
from datetime import datetime

import torch
import numpy as np
import h5py
import argparse
from torch import nn
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from torchvision import transforms
import wandb
import pybullet as p

from .constants import JOINT_ORDER

# Add parent directory to Python path (assuming act_relevant_files is in mega_rl_experiments/)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add act_relevant_files directory to path for util module
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'act_relevant_files'))

# Instead of defining SequencePolicy, we use ACTPolicy
from act_relevant_files.policy import ACTPolicy
from .pybullet_utils import set_joint_angles_instantly, get_dummy_image


# Instead of using dataset directly, let's define constants
ACTION_MEAN = 0.0  # You'll need to set these to your actual values
ACTION_STD = 1.0   # You'll need to set these to your actual values

# Update denormalization to use constants
def denormalize(data):
    return data * ACTION_STD + ACTION_MEAN


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
    {'l_ank_roll': 0.0, 'r_ank_roll': 0.0, 'l_ank_pitch': 0.5864306186666667, 'r_ank_pitch': -0.5864306186666667, 'l_knee': 1.0890854346666667, 'r_knee': -1.0890854346666667, 'l_hip_pitch': -0.7120943226666667, 'r_hip_pitch': 0.7120943226666667, 'l_hip_roll': 0.0, 'r_hip_roll': 0.0, 'l_hip_yaw': 0.0, 'r_hip_yaw': 0.0, 'l_sho_pitch': 0.16755160533333335, 'r_sho_pitch': -0.16755160533333335, 'l_sho_roll': 1.3823007440000001, 'r_sho_roll': -1.3823007440000001, 'l_el_pitch': 0.0, 'r_el_pitch': 0.0, 'l_el_yaw': -1.8011797573333335, 'r_el_yaw': 1.8011797573333335, 'l_gripper': 0.0, 'r_gripper': 0.0}
]

# Modified dataset class for sequence prediction
class ServoDataset(Dataset):
    """Dataset for sequence prediction with sliding window"""
    def __init__(self, action_sequences, num_samples=1000, window_size=3, image_size=64,
                 use_real_images=False):
        # Add these as instance variables if you need them
        self.action_mean = ACTION_MEAN
        self.action_std = ACTION_STD
        # Convert each sequence from list of dicts to list of tensors
        self.action_sequences = [
            [self.dict_to_tensor(step) for step in seq]
            for seq in action_sequences
        ]
        self.window_size = window_size
        self.images = torch.rand(num_samples, 1, 3, image_size, image_size) if use_real_images else torch.zeros(num_samples, 1, 3, image_size, image_size)

        # Create sequence targets with proper windowing
        self.targets = []
        for _ in range(num_samples):
            seq_idx = np.random.randint(0, len(self.action_sequences))
            seq = self.action_sequences[seq_idx]
            start = np.random.randint(0, len(seq) - self.window_size + 1)
            window = seq[start:start + self.window_size]
            self.targets.append(torch.stack(window))

        # Add data augmentation configuration
        self.use_real_images = use_real_images
        self.augment = transforms.Compose([
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5),
            transforms.RandomAdjustSharpness(2, p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ])


    def dict_to_tensor(self, action_dict):
        return torch.tensor([
            action_dict.get(name, 0.0)  # Use 0.0 as default for missing joints
            for name in JOINT_ORDER
        ], dtype=torch.float32)


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        """Returns a single sample from the dataset."""
        image = self.images[idx]  # Now shape is [1, 3, H, W]

        # Only apply augmentations if image isn't black or allowed explicitly
        if not self.use_real_images and torch.all(image == 0):
            pass
        else:
            # Apply augmentations to each camera view separately
            image = image.squeeze(0)  # [3, H, W]
            image = self.augment(image)
            image = image.unsqueeze(0)  # Back to [1, 3, H, W]

        return image, torch.zeros(24), self.targets[idx]


def train(policy, train_loader, num_epochs=50, lr=1e-4, device='cpu', policy_config=None, args=None):
    scaler = torch.cuda.amp.GradScaler()  # Add mixed precision
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr, weight_decay=1e-5)
    # More balanced weighting for problem joints
    loss_weights = torch.ones(24)
    loss_weights[[
        JOINT_ORDER.index('r_hip_yaw'),
        JOINT_ORDER.index('r_ank_pitch'),
        JOINT_ORDER.index('r_el_yaw')
    ]] = 2.0
    criterion = nn.SmoothL1Loss(reduction='none')  # Remove weight parameter
    best_loss = float('inf')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=3, factor=0.3, verbose=True
    )

    total_start_time = time.time()

    # Add validation frequency parameters
    GREET_VAL_FREQ = 5  # Validate greet sequence every 5 epochs
    PYBLET_VAL_FREQ = 10  # Run PyBullet eval every 10 epochs
    best_greet_error = float('inf')

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        policy.train()
        total_loss = 0  # Reset each epoch
        joint_errors = {name: 0.0 for name in JOINT_ORDER}

        for batch_idx, (images, qpos, targets) in enumerate(train_loader):
            images = images.to(device)
            qpos = qpos.to(device)
            targets = targets.to(device)  # shape: [B, window_size, 24]

            with torch.cuda.amp.autocast():
                # Reshape targets to match policy output
                current_targets = targets[:, 0, :]  # Only predict first timestep for now
                preds = policy(qpos, images)  # shape: [B, 24]
                loss = (criterion(preds, current_targets) * loss_weights.to(device)).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            # Calculate per-joint errors
            with torch.no_grad():
                errors = torch.abs(preds - current_targets).mean(dim=0)  # Average over batch
                for i, name in enumerate(JOINT_ORDER):
                    joint_errors[name] += errors[i].item()

            # Log batch metrics
            wandb.log({
                "batch_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch_progress": epoch + (batch_idx+1)/len(train_loader)
            })

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints', exist_ok=True)  # Create directory if needed
            ckpt_path = os.path.join('checkpoints', f'policy_epoch{epoch}_{timestamp}.pth')
            torch.save({
                'model_state': policy.state_dict(),
                'config': policy_config,  # Add config to checkpoint
                'training_args': vars(args)
            }, ckpt_path)

        # Update learning rate
        scheduler.step(total_loss / len(train_loader))

        # Print diagnostics - modified to show problematic joints
        avg_loss = total_loss / len(train_loader)
        print(f'\nEpoch {epoch}: Loss: {avg_loss:.6f}')

        # Print per-joint errors
        print("\nAverage Joint Errors:")
        for name in JOINT_ORDER[:5] + ['r_el_yaw', 'l_el_yaw']:  # Focus on key problem joints
            avg_error = joint_errors[name] / len(train_loader)
            print(f"  {name:15}: {avg_error:.5f}")

        # Log denormalized errors to WandB
        wandb.log({
            "epoch_loss": avg_loss,
            **{f"err_{name}": denormalize(torch.tensor(joint_errors[name]/len(train_loader))).item()
               for name in JOINT_ORDER[:5] + ['r_el_yaw', 'l_el_yaw']}
        })

        # Log sample predictions
        with torch.no_grad():
            sample_img, sample_qpos, sample_target = next(iter(train_loader))
            # Ensure sample_img has correct shape [B, num_cam, C, H, W]
            if sample_img.dim() == 4:
                sample_img = sample_img.unsqueeze(1)
            if sample_img.shape[2] == 1:
                sample_img = sample_img.repeat(1, 1, 3, 1, 1)
            sample_pred = preds[0].cpu()  # Just take first batch item
            sample_target = current_targets[0].cpu()  # First batch item of current timestep

            # Denormalize before logging
            sample_pred_denorm = denormalize(sample_pred)
            sample_target_denorm = denormalize(sample_target)

            # Log denormalized values
            print("\nSample prediction vs target (DENORMALIZED):")
            for i, name in enumerate(JOINT_ORDER[:5] + ['r_el_yaw', 'l_el_yaw']):
                pred_val = sample_pred_denorm[i].item()
                target_val = sample_target_denorm[i].item()
                print(f"  {name.ljust(12)}: Pred: {pred_val:.5f} vs Target: {target_val:.5f} (err: {abs(pred_val-target_val):.5f})")

            # Log to wandb without using numpy
            wandb.log({
                "sample_prediction": wandb.Histogram(sample_pred.tolist()),
                "sample_target": wandb.Histogram(sample_target.tolist())
            })

        # Update best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state': policy.state_dict(),
                'config': policy_config,  # Add config to checkpoint
                'training_args': vars(args)
            }, os.path.join(os.path.dirname(__file__), f'best_policy_{timestamp}.pth'))

        # After completing epoch training:
        if epoch % GREET_VAL_FREQ == 0:
            greet_mse, greet_max_err = validate_greet_sequence(policy, device)
            wandb.log({
                "greet_mse": greet_mse,
                "greet_max_error": greet_max_err
            })

            # Update best greet error
            if greet_mse < best_greet_error:
                best_greet_error = greet_mse
                torch.save(policy.state_dict(), f'best_greet_policy_{timestamp}.pth')

        if epoch % PYBLET_VAL_FREQ == 0:
            print("\nRunning PyBullet evaluation...")
            try:
                sim_metrics = evaluate_in_pybullet(policy, device)
                wandb.log({
                    "com_deviation": np.mean(sim_metrics['com_deviation']),
                    "max_com_deviation": np.max(sim_metrics['com_deviation']),
                    "fall_detected": float(sim_metrics['fall_detected'])
                })
                print(f"PyBullet Metrics: COM Dev {np.mean(sim_metrics['com_deviation']):.3f}m, Fall: {sim_metrics['fall_detected']}")
            except Exception as e:
                print(f"PyBullet evaluation failed: {str(e)}")
            # torch.save(policy.state_dict(), os.path.join(os.path.dirname(__file__), 'checkpoints', f'johnny_imitate_best_policy_{timestamp}.pth'))

        epoch_time = time.time() - epoch_start_time
        mins, secs = divmod(epoch_time, 60)
        print(f'Epoch {epoch} took: {int(mins)}m {secs:.1f}s | Loss: {avg_loss:.7f}')

    total_time = time.time() - total_start_time
    hours, rem = divmod(total_time, 3600)
    mins, secs = divmod(rem, 60)
    print(f'\nTotal training time: {int(hours):02d}h {int(mins):02d}m {secs:.1f}s')


def validate_greet_sequence(policy, device='cpu'):
    """Evaluate policy on known GREET sequence without PyBullet"""
    # TODO untested
    policy.eval()
    greet_targets = torch.stack([torch.tensor([step[name] for name in JOINT_ORDER])
                               for step in all_greet_action_lines])

    with torch.no_grad():
        # Create dummy inputs matching training format
        dummy_images = torch.zeros(len(greet_targets), 3, 64, 64).to(device)
        dummy_qpos = torch.zeros(len(greet_targets), 24).to(device)

        # Predict full sequence
        preds = policy(dummy_qpos, dummy_images)

    # Calculate metrics
    mse = torch.mean((preds - greet_targets.to(device))**2).item()
    max_error = torch.max(torch.abs(preds - greet_targets.to(device))).item()

    print(f"\nGREET Sequence Validation:")
    print(f"  MSE: {mse:.5f}")
    print(f"  Max Joint Error: {max_error:.5f} rad")
    return mse, max_error


def evaluate_in_pybullet(policy, device='cpu'):
    """Run policy in headless PyBullet simulation"""
    p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    # Use the same URDF path as your other scripts
    robot = p.loadURDF("/home/ben/all_projects/ainex_private_ws/ainex_private/src/ainex_simulations/ainex_description/urdf/ainex.urdf",
                      [0, 0, 0.25],
                      useFixedBase=False)  # Match your training setup

    metrics = {
        'com_deviation': [],
        'joint_errors': [],
        'fall_detected': False
    }

    for step in range(100):  # Simulate 100 steps
        image = get_dummy_image().to(device)
        qpos = torch.zeros(1, 24).to(device)

        with torch.no_grad():
            action = policy(qpos, image).cpu().numpy().flatten()

        # Apply action
        joint_targets = {name: action[i] for i, name in enumerate(JOINT_ORDER)}
        set_joint_angles_instantly(robot, joint_targets)
        p.stepSimulation()

        # Calculate metrics
        com_pos, _ = p.getCenterOfMass(robot)
        metrics['com_deviation'].append(np.linalg.norm(com_pos[:2]))  # XY plane deviation

        # Check for fall
        if com_pos[2] < 0.1:  # Adjust based on robot height
            metrics['fall_detected'] = True
            break

    p.disconnect()
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='',  # Keep argument but don't use it
                      help='Not used, preserved for compatibility')
    parser.add_argument('--num_epochs', type=int, default=500,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                      help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                      help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu',
                      choices=['cpu', 'cuda'],
                      help='Device to train on (cpu or cuda)')
    parser.add_argument('--use_real_images', action='store_true',
                      help='Use real images instead of black, enables augmentation')
    parser.add_argument('--wandb_project', type=str, default='imitate_johnny_act',
                      help='Weights & Biases project name')
    args = parser.parse_args()

    # Set device
    device = torch.device(args.device)
    print(f"Training on {device}")

    # Create synthetic dataset
    greet_sequences = [all_greet_action_lines]  # Can add more sequences
    dataset = ServoDataset(
        action_sequences=greet_sequences,
        num_samples=5000,
        use_real_images=args.use_real_images,
        window_size=5
    )
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Configure ACTPolicy to mimic the original SequencePolicy behavior
    policy_config = {
        'num_queries': 1,
        'kl_weight': 1,
        'task_name': 'imitate_johnny_act',
        'device': args.device,
        'num_actions': 24,
        'state_dim': 24,
        'hidden_dim': 512,
        'dim_feedforward': 3200,
        'lr_backbone': 1e-5,
        'backbone': 'resnet18',
        'enc_layers': 2,
        'dec_layers': 2,
        'nheads': 8,
        'dropout': 0.1,
        'camera_names': ['dummy'],
    }

    policy = ACTPolicy(policy_config).to(device)

    # Initialize WandB
    wandb.init(
        project=args.wandb_project,
        config={
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "hidden_dim": policy_config['hidden_dim'],
            "dim_feedforward": policy_config['dim_feedforward'],
            "window_size": dataset.window_size,
            "use_real_images": args.use_real_images
        }
    )

    train(policy, train_loader, num_epochs=args.num_epochs, lr=args.lr, device=device, policy_config=policy_config, args=args)

    # Save trained policy with timestamp and epochs
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'servo_policy_24dof_{timestamp}_ep{args.num_epochs}.pth'
    torch.save({
        'model_state': policy.state_dict(),
        'config': policy_config,  # Add config to checkpoint
        'training_args': vars(args)
    }, save_path)


if __name__ == '__main__':
    main()

"""
python imitate_johnny_actions/imitate_johnny_action_act.py --device cuda
"""
