import os
import time
import numpy as np
import torch
import cv2
import pyautogui
from torch import nn
from torch.utils.data import Dataset, DataLoader
from collections import deque
import h5py
from datetime import datetime

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

class MousePolicy(nn.Module):
    """CNN that predicts mouse position from screen content"""
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2),  # (240,240) -> (119,119)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), # (119,119) -> (59,59)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*59*59, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # Output x,y coordinates
        )
        
    def forward(self, x):
        return self.cnn(x)

class MouseDataset(Dataset):
    def __init__(self, recordings, image_size=240):
        self.images = torch.stack([torch.from_numpy(x).float()/255.0 
                                 for x in recordings['images']])
        self.positions = torch.tensor(recordings['positions'], dtype=torch.float32)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx].permute(2, 0, 1), self.positions[idx]

def train_mouse_policy(data_path='mouse_demo.hdf5', num_epochs=50):
    # Load recorded data
    with h5py.File(data_path, 'r') as f:
        recordings = {
            'images': f['images'][:],
            'positions': f['positions'][:]
        }
    
    dataset = MouseDataset(recordings)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model and training
    policy = MousePolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for images, targets in loader:
            optimizer.zero_grad()
            outputs = policy(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch} Loss: {total_loss/len(loader):.4f}")
    
    # Save trained model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(policy.state_dict(), f"mouse_policy_{timestamp}.pth")

def run_policy(policy_path):
    """Run trained policy to generate mouse movements"""
    policy = MousePolicy()
    policy.load_state_dict(torch.load(policy_path))
    policy.eval()
    
    recorder = MouseRecorder()
    
    try:
        while True:
            # Capture screen
            recorder.capture_frame()
            if len(recorder.history) < recorder.history.maxlen:
                continue
                
            # Prepare input tensor
            current_frame = np.stack(recorder.history)
            input_tensor = torch.from_numpy(current_frame).float()/255.0
            input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)
            
            # Get prediction
            with torch.no_grad():
                pred = policy(input_tensor)[0].numpy()
                
            # Move mouse
            pyautogui.moveTo(pred[0], pred[1], duration=0.01)
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("Stopping policy execution")

if __name__ == "__main__":
    # To record demonstration:
    demo_data = circular_mouse_controller(duration=60)
    with h5py.File('mouse_demo.hdf5', 'w') as f:
        f.create_dataset('images', data=demo_data['images'])
        f.create_dataset('positions', data=demo_data['positions'])
    
    # To train:
    train_mouse_policy(num_epochs=50)
    
    # To run (use latest policy):
    # run_policy("mouse_policy_20240315_123456.pth")
