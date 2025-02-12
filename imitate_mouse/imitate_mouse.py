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
        
        # Add dummy qpos (14 dim like robot state)
        self.qpos = torch.zeros((len(recordings['positions']), 14))  
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        frames = self.images[idx]
        merged = frames.permute(0, 3, 1, 2)  # [T, C, H, W]
        return merged, self.qpos[idx], self.positions[idx], torch.zeros(1)  # image, qpos, action, is_pad


def train_mouse_policy(data_path='mouse_demo.hdf5', num_epochs=50):
    with h5py.File(data_path, 'r') as f:
        recordings = {'images': f['images'][:], 'positions': f['positions'][:]}
    
    dataset = MouseACTDataset(recordings)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # ACT policy config
    policy_config = {
        'lr': 1e-5,
        'num_queries': 5,  # Predict 5 steps ahead
        'kl_weight': 10,
        'hidden_dim': 512,
        'dim_feedforward': 3200,
        'lr_backbone': 1e-5,
        'backbone': 'resnet18',
        'enc_layers': 4,
        'dec_layers': 7,
        'nheads': 8,
        'camera_names': ['mouse_cam'],
    }
    
    policy = ACTPolicy(policy_config)
    optimizer = policy.configure_optimizers()
    
    # Training loop adapted from imitate_episodes.py
    best_loss = float('inf')
    for epoch in range(num_epochs):
        policy.train()
        total_loss = 0
        
        for images, qpos, actions, is_pad in loader:
            images = images.cuda()
            qpos = qpos.cuda()
            actions = actions.cuda()
            is_pad = is_pad.cuda()
            
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


def run_policy(policy_path):
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
    })
    policy.load_state_dict(torch.load(policy_path))
    policy.cuda().eval()
    
    recorder = MouseRecorder()
    recorder.start_recording()
    
    try:
        while True:
            recorder.capture_frame()
            if len(recorder.history) < recorder.history.maxlen:
                continue
                
            current_frame = np.stack(recorder.history)
            input_tensor = torch.from_numpy(current_frame).float()/255.0
            input_tensor = input_tensor.permute(0, 3, 1, 2).cuda()  # [T, C, H, W]
            
            # Generate dummy qpos
            qpos = torch.zeros(14).cuda()
            
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
    # # To record demonstration:
    # demo_data = circular_mouse_controller(duration=60)
    # with h5py.File('mouse_demo.hdf5', 'w') as f:
    #     f.create_dataset('images', data=demo_data['images'])
    #     f.create_dataset('positions', data=demo_data['positions'])
    
    # To train:
    # train_mouse_policy(num_epochs=50)
    
    # To run (use latest policy):
    run_policy("mouse_act_policy_best.ckpt")