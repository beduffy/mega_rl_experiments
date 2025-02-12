import os
import time
import sys
from datetime import datetime
from collections import deque

import numpy as np
import torch
import cv2
import pyautogui
from torch import nn
from torch.utils.data import Dataset, DataLoader
import h5py
import argparse

path_to_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(path_to_root)
sys.path.append(path_to_root)
from act_relevant_files.policy import ACTPolicy
from act_relevant_files.utils import load_data, compute_dict_mean


class MouseRecorder:
    """Records mouse movements and screen content"""
    def __init__(self, screen_region=(0, 0, 1920, 1080), history_length=3):
        self.screen_region = screen_region
        self.history = deque(maxlen=history_length)
        self.recording = False
        self.data = {
            'images': [],
            'positions': [],
            'timestamps': []
        }
        
    def start_recording(self):
        self.recording = True
        self.data = {'images': [], 'positions': [], 'timestamps': []}
        
    def stop_recording(self):
        self.recording = False
        
    def capture_frame(self):
        if not self.recording:
            return
            
        # Capture screen
        img = pyautogui.screenshot(region=self.screen_region)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (240, 240))
        
        # Store in history
        self.history.append(img)
        
        # Only save when history is full
        if len(self.history) == self.history.maxlen:
            self.data['images'].append(np.stack(self.history))
            self.data['positions'].append(pyautogui.position())
            self.data['timestamps'].append(time.time())

def circular_mouse_controller(radius=300, speed=2, duration=10):
    """Scripted mouse controller that moves in circles"""
    recorder = MouseRecorder()
    recorder.start_recording()
    
    start_time = time.time()
    center_x, center_y = pyautogui.size().width//2, pyautogui.size().height//2
    
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

class MouseACTDataset(Dataset):
    def __init__(self, recordings, image_size=240, screen_size=(1920, 1080)):
        self.images = torch.from_numpy(np.array(recordings['images'])).float() / 255.0
        self.positions = torch.tensor(recordings['positions'], dtype=torch.float32)
        self.positions[:, 0] /= screen_size[0]  # Normalize X
        self.positions[:, 1] /= screen_size[1]  # Normalize Y
        
        # Use normalized mouse positions as the qpos (2D state) instead of a 14-dim dummy.
        self.qpos = self.positions.clone()
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        frames = self.images[idx]
        merged = frames.permute(0, 3, 1, 2)  # [T, C, H, W]
        return merged, self.qpos[idx], self.positions[idx], torch.zeros(1, dtype=torch.bool)  # image, qpos, action, is_pad


def train_mouse_policy(args_dict, device='cuda'):
    with h5py.File('mouse_demo.hdf5', 'r') as f:
        recordings = {'images': f['images'][:], 'positions': f['positions'][:]}
    
    dataset = MouseACTDataset(recordings)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
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
        
        for images, qpos, actions, is_pad in loader:
            images = images.to(device)
            qpos = qpos.to(device)

            # qpos = torch.zeros(1, 2).to(device)  # Add batch dimension (1,2)
            # qpos = qpos.repeat(images.size(0), 1)  # Expand to match batch size
            # actions = torch.stack([torch.tensor([x/1920, y/1080]) for x, y in actions])
            actions = actions.to(device)
            is_pad = is_pad.to(device)
            
            # Forward pass
            loss_dict = policy(qpos, images, actions, is_pad)
            loss = loss_dict['loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        print(f'Epoch {epoch} Loss: {avg_loss:.4f}')
        
        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(policy.state_dict(), f"mouse_act_policy_best.ckpt")


def run_policy(policy_path, device='cuda'):
    policy = ACTPolicy({
        'num_queries': 5,
        'hidden_dim': 512,
        'dim_feedforward': 3200,
        'lr_backbone': 1e-5,
        'backbone': 'resnet18',
        'enc_layers': 4,
        'dec_layers': 7,
        'nheads': 8,
        'camera_names': ['mouse_cam'],
        'kl_weight': 10
    }).to(device)
    policy.load_state_dict(torch.load(policy_path))
    policy.eval()
    
    recorder = MouseRecorder()
    recorder.start_recording()
    
    try:
        while True:
            recorder.capture_frame()
            if len(recorder.history) < recorder.history.maxlen:
                continue
                
            current_frame = np.stack(recorder.history)
            input_tensor = torch.from_numpy(current_frame).float()/255.0
            input_tensor = input_tensor.permute(0, 3, 1, 2).to(device)  # [T, C, H, W]
            
            # Generate dummy qpos
            qpos = torch.zeros(14).to(device)
            
            # ACT inference
            with torch.no_grad():
                action = policy(qpos, input_tensor.unsqueeze(0))
                pred_normalized = action[0,0].cpu().numpy()  # Take first predicted action
                
            pred_x = int(pred_normalized[0] * 1920)
            pred_y = int(pred_normalized[1] * 1080)
            pyautogui.moveTo(pred_x, pred_y, duration=0.01)
            
    except KeyboardInterrupt:
        recorder.stop_recording()


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
    args = parser.parse_args(remaining_args)
    
    # Pass args to training
    train_mouse_policy(
        args_dict=vars(args),
        device=device
    )

    """
    python3 imitate_mouse.py --task_name sim_transfer_cube_scripted --ckpt_dir checkpoints --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000  --lr 1e-5 --seed 0
    """