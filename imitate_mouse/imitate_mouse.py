import os
import time
import sys
from datetime import datetime
from collections import deque

import numpy as np
import torch
import cv2
import h5py
import argparse
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb

path_to_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(path_to_root)
sys.path.append(path_to_root)
from act_relevant_files.policy import ACTPolicy
from act_relevant_files.utils import load_data, compute_dict_mean


class MouseRecorder:
    """Records mouse movements and screen content"""
    def __init__(self, screen_region=(0, 0, 1920, 1080), history_length=3, use_dummy=False):
        self.screen_region = screen_region
        self.history = deque(maxlen=history_length)
        self.recording = False
        self.data = {
            'images': [],
            'positions': [],
            'timestamps': []
        }
        self.use_dummy = use_dummy
        self.dummy_size = (64, 64)  # Smaller dummy images
        self.dummy_pos = (960, 540)  # Center position for dummy mode
        
    def start_recording(self):
        self.recording = True
        self.data = {'images': [], 'positions': [], 'timestamps': []}
        
    def stop_recording(self):
        self.recording = False
        
    def capture_frame(self):
        if not self.recording:
            return
            
        if self.use_dummy:
            img = np.zeros((*self.dummy_size, 3), dtype=np.uint8)
            pos = self.dummy_pos
        else:
            try:
                import pyautogui  # Moved import inside conditional
                img = pyautogui.screenshot(region=self.screen_region)
                pos = pyautogui.position()
            except Exception as e:
                print(f"GUI access failed: {str(e)}")
                return

        # Store in history
        self.history.append(img)
        
        # Only save when history is full
        if len(self.history) == self.history.maxlen:
            self.data['images'].append(np.stack(self.history))
            self.data['positions'].append(pos)
            self.data['timestamps'].append(time.time())

def circular_mouse_controller(radius=300, speed=2, duration=10, use_dummy=False):
    """Scripted mouse controller that moves in circles"""
    if not use_dummy:
        try:
            import pyautogui  # Ensure import stays inside conditional
            center_x, center_y = pyautogui.position()
            recorder = MouseRecorder(use_dummy=use_dummy)
            recorder.start_recording()
            
            start_time = time.time()
            
            try:
                while time.time() - start_time < duration:
                    angle = (time.time() - start_time) * speed
                    x = center_x + int(radius * np.cos(angle))
                    y = center_y + int(radius * np.sin(angle))
                    pyautogui.moveTo(x, y, duration=0.01)
                    recorder.capture_frame()
                    time.sleep(0.01)
            finally:
                recorder.stop_recording()
                return recorder.data
        except ImportError:
            print("PyAutoGUI not available in headless mode")
    else:
        print("Dummy mode: Skipping actual mouse movements")

class MouseACTDataset(Dataset):
    def __init__(self, recordings, image_size=64, screen_size=(1920, 1080)):
        # Add resize transform
        self.resize = transforms.Resize((image_size, image_size))
        
        # Fix dummy image dimensions
        if isinstance(recordings['images'], np.ndarray) and recordings['images'].dtype == np.float64:
            recordings['images'] = recordings['images'].astype(np.float32)
            
        self.images = torch.stack([
            self.resize(torch.tensor(img).float().permute(0, 3, 1, 2))  # Now using defined resize
            for img in recordings['images']
        ]) / 255.0
        self.positions = torch.tensor(recordings['positions'], dtype=torch.float32)
        self.positions[:, 0] /= screen_size[0]  # Normalize X
        self.positions[:, 1] /= screen_size[1]  # Normalize Y
        
        # Use normalized mouse positions as the qpos (2D state) instead of a 14-dim dummy.
        self.qpos = self.positions.clone()
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        frames = self.images[idx]
        return frames, self.qpos[idx], self.positions[idx], torch.zeros(1, dtype=torch.bool)


def train_mouse_policy(args_dict, device='cuda'):
    # Initialize WandB with hyperparameters
    wandb.init(
        project="imitate_mouse",
        config={
            "policy_class": args_dict['policy_class'],
            "kl_weight": args_dict['kl_weight'],
            "chunk_size": args_dict['chunk_size'],
            "hidden_dim": args_dict['hidden_dim'],
            "batch_size": args_dict['batch_size'],
            "dim_feedforward": args_dict['dim_feedforward'],
            "num_epochs": args_dict['num_epochs'],
            "lr": args_dict['lr'],
            "seed": args_dict['seed'],
            "use_dummy": args_dict.get('use_dummy_images', False)
        }
    )
    
    # Generate complete dummy dataset if needed
    if args_dict.get('use_dummy_images'):
        num_samples = 1000
        timesteps = 3
        recordings = {
            'images': np.zeros((num_samples, timesteps, 64, 64, 3), dtype=np.uint8),
            'positions': np.random.rand(num_samples, 2).astype(np.float32)
        }
    else:
        # Original loading code
        with h5py.File('mouse_demo.hdf5', 'r') as f:
            recordings = {'images': f['images'][:], 'positions': f['positions'][:]}
    
    # Optional: Replace with black frames
    if args_dict.get('use_dummy_images'):
        # Create dummy data with correct dimensions (num_samples, timesteps, height, width, channels)
        dummy_images = np.zeros((len(recordings['images']), 3, 64, 64, 3), dtype=np.uint8)  # 3 timesteps
        recordings['images'] = dummy_images
    
    dataset = MouseACTDataset(recordings)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2, persistent_workers=True)
    
    # ACT policy config
    policy_config = {
        'lr': args_dict['lr'],
        'num_queries': args_dict['chunk_size'],
        'kl_weight': args_dict['kl_weight'],
        'hidden_dim': args_dict['hidden_dim'],
        'dim_feedforward': args_dict['dim_feedforward'],
        'lr_backbone': 1e-5,
        'backbone': 'resnet18',
        'enc_layers': 4,
        'dec_layers': 7,
        'nheads': 8,
        'num_actions': 2,
        'state_dim': 2,
        'camera_names': ['mouse_cam'],
    }
    
    policy = ACTPolicy(policy_config).to(device)
    optimizer = policy.configure_optimizers()
    
    # Training loop adapted from imitate_episodes.py
    best_loss = float('inf')
    for epoch in range(args_dict['num_epochs']):
        policy.train()
        total_loss = 0
        total_x_error = 0.0
        total_y_error = 0.0
        
        epoch_start = time.time()
        
        progress = tqdm(loader, desc=f'Epoch {epoch}', unit='batch')
        for batch_idx, (images, qpos, actions, is_pad) in enumerate(progress):
            images = images.to(device)
            qpos = qpos.to(device)
            actions = actions.to(device)
            is_pad = is_pad.to(device)
            
            # Forward pass
            loss_dict = policy(qpos, images, actions, is_pad)
            loss = loss_dict['loss']
            
            # Accumulate errors
            with torch.no_grad():
                # Get action predictions directly from model's forward pass
                a_hat = policy.model(qpos, images, None, actions, is_pad)[0]  # Access first element of tuple
                pred_denorm = a_hat.detach().cpu()
                target_denorm = actions.detach().cpu()
                x_error = torch.abs(pred_denorm[:,0] - target_denorm[:,0]).mean().item()
                y_error = torch.abs(pred_denorm[:,1] - target_denorm[:,1]).mean().item()
                total_x_error += x_error * images.size(0)
                total_y_error += y_error * images.size(0)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress.set_postfix({
                'loss': f'{loss.item():.4f}',
                'x_err': f'{x_error:.4f}',
                'y_err': f'{y_error:.4f}'
            })
            
            # Enhanced logging
            wandb.log({
                "loss": loss.item(),
                "x_error": x_error,
                "y_error": y_error,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + (batch_idx+1)/len(loader),  # Smooth epoch progress
                "predictions": wandb.Histogram(pred_denorm.numpy()),
                "targets": wandb.Histogram(target_denorm.numpy()),
                "sample_prediction_x": pred_denorm[0,0].item(),
                "sample_prediction_y": pred_denorm[0,1].item(),
                "sample_target_x": target_denorm[0,0].item(),
                "sample_target_y": target_denorm[0,1].item()
            })

            # Inside training loop once per epoch
            # wandb.log({"sample_images": [wandb.Image(img) for img in images[0].cpu().numpy()]})
        
        epoch_time = time.time() - epoch_start
        
        # Print diagnostics with 8 decimal places
        avg_loss = total_loss / len(loader)
        avg_x_error = total_x_error / len(dataset)
        avg_y_error = total_y_error / len(dataset)
        print(f'\nEpoch {epoch} Loss: {avg_loss:.8f} (took {epoch_time:.2f}s)')
        print(f'X Error: {avg_x_error:.8f}px | Y Error: {avg_y_error:.8f}px')
        
        # Print sample predictions
        with torch.no_grad():
            sample_images, sample_qpos, sample_actions, sample_is_pad = next(iter(loader))
            sample_images = sample_images.to(device)
            sample_qpos = sample_qpos.to(device)
            
            # Direct model call for inference
            a_hat, _, _ = policy.model(sample_qpos, sample_images, env_state=None)
            sample_pred = a_hat.cpu()
            sample_target = sample_actions.cpu()
            
            print("\nSample predictions vs targets:")
            for i in range(3):  # First 3 samples
                pred_x, pred_y = sample_pred[i].numpy()
                target_x, target_y = sample_target[i].numpy()
                print(f"  Sample {i}:")
                print(f"    Predicted: ({pred_x:.8f}, {pred_y:.8f})")
                print(f"    Target:    ({target_x:.8f}, {target_y:.8f})")
                print(f"    Error:     ({abs(pred_x-target_x):.8f}, {abs(pred_y-target_y):.8f})px")

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'mouse_act_policy_best.ckpt')
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(policy.state_dict(), checkpoint_path)

        # Log epoch-level metrics
        wandb.log({
            "epoch_loss": avg_loss,
            "epoch_x_error": avg_x_error,
            "epoch_y_error": avg_y_error,
            "epoch_time": epoch_time
        })


if __name__ == "__main__":
    # Separate device arg first
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      choices=['cpu', 'cuda'], help='Device to run policy on')
    args, remaining_args = parser.parse_known_args()
    device = args.device

    # Now parse ACT/DETR args separately
    parser = argparse.ArgumentParser()
    # Add ACT-specific args from imitate_episodes.py
    parser.add_argument('--task_name', type=str, required=True, help='Task name')
    parser.add_argument('--ckpt_dir', type=str, required=True, help='Checkpoint directory')
    parser.add_argument('--policy_class', type=str, required=True, choices=['ACT', 'CNNMLP'], help='Policy class')
    parser.add_argument('--kl_weight', type=int, required=True, help='KL weight')
    parser.add_argument('--chunk_size', type=int, required=True, help='Chunk size')
    parser.add_argument('--hidden_dim', type=int, required=True, help='Hidden dimension')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('--dim_feedforward', type=int, required=True, help='Feedforward dimension')
    parser.add_argument('--num_epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    parser.add_argument('--use_dummy_images', action='store_true', help='Use dummy images for training')
    args = parser.parse_args(remaining_args)
    
    # Pass args to training
    train_mouse_policy(
        args_dict=vars(args),
        device=device
    )

    """
    python3 imitate_mouse.py --task_name sim_transfer_cube_scripted --ckpt_dir checkpoints --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000  --lr 1e-5 --seed 0 --use_dummy_images
    python3 imitate_mouse/imitate_mouse.py --task_name sim_transfer_cube_scripted --ckpt_dir checkpoints --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000  --lr 1e-5 --seed 0 --use_dummy_images --device cuda
    """