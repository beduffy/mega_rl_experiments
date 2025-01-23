import torch
import numpy as np
import h5py
import os
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2


class SimplePolicy(nn.Module):
    """Neural network that maps images (and optionally qpos) to predicted values.
    
    Input shapes:
        images: (batch_size, 3, 240, 240) - RGB images normalized to [0,1]
        qpos: (batch_size, qpos_dim) - Optional state information
        
    Output shape:
        predictions: (batch_size, 1) - Single predicted value per input
        
    Example for batch_size=32:
        Input image batch shape: (32, 3, 240, 240)
        Input qpos batch shape: (32, 2)
        Output shape: (32, 1)
    """
    def __init__(self, image_size=240, use_qpos=False, qpos_dim=0):
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

        # MLP combines flattened image features with qpos if present
        combined_input_size = flat_size + (qpos_dim if use_qpos else 0)
        self.mlp = nn.Sequential(
            nn.Linear(combined_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Final output: single predicted value
        )
        
        self.use_qpos = use_qpos
    
    def forward(self, images, qpos=None):
        x = self.conv(images)
        if self.use_qpos:
            x = torch.cat([x, qpos], dim=1)
        return self.mlp(x)


class DemoDataset(Dataset):
    """Dataset that loads and provides demonstration data.
    
    Data structure in HDF5 file:
        /observations/
            images/
                main: (num_frames, 240, 240, 3) - RGB images
            qpos: (num_frames, qpos_dim) - Optional state information
        /action: (num_frames, 1) - Target values to predict
        
    Example of data loading:
        dataset = DemoDataset('demo_data.hdf5')
        # If dataset has qpos:
        image, qpos, target = dataset[0]  # Single sample
        # If dataset has no qpos:
        image, target = dataset[0]  # Single sample
        
    Example shapes for a single sample:
        image: (3, 240, 240)
        qpos: (2,)  # If present
        target: (1,)
    """
    def __init__(self, data_path):
        with h5py.File(data_path, 'r') as f:
            # Load and convert to torch tensors
            self.images = torch.from_numpy(f['/observations/images/main'][:]).float() / 255.0
            self.targets = torch.from_numpy(f['/action'][:, 0]).float()
            # Optional qpos data
            self.qpos = (torch.from_numpy(f['/observations/qpos'][:]).float() 
                        if '/observations/qpos' in f else None)
            
        # Rearrange image dimensions from (N,H,W,C) to (N,C,H,W) for PyTorch
        self.images = self.images.permute(0, 3, 1, 2)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """Returns a single sample from the dataset.
        
        For temporal dataset, consecutive indices represent temporal sequence:
        idx:     0  1  2  3  4  5  6  7  8  9 ...
        target:  1  2  3  4  5  6  7  8  9  8 ...
        """
        if self.qpos is not None:
            return self.images[idx], self.qpos[idx], self.targets[idx]
        return self.images[idx], self.targets[idx]


def create_demo_data_fixed(output_path, num_frames=100, target_value=5.0):
    """Create simple demonstration data - random images with fixed target value"""
    images = np.random.randint(0, 255, (num_frames, 240, 240, 3), dtype=np.uint8)
    actions = np.full((num_frames, 1), target_value, dtype=np.float32)
    
    with h5py.File(output_path, 'w') as f:
        f.attrs['sim'] = True
        f.create_dataset('/observations/images/main', data=images)
        f.create_dataset('/action', data=actions)


def create_demo_data_black_white(output_path, num_frames=100):
    """Create demonstration data where black images map to 5.0 and white images map to 0.0"""
    images = []
    actions = []
    
    for i in range(num_frames):
        if i < num_frames // 2:
            img = np.zeros((240, 240, 3), dtype=np.uint8)
            target = 5.0
        else:
            img = np.full((240, 240, 3), 255, dtype=np.uint8)
            target = 0.0
        
        images.append(img)
        actions.append([target])
    
    images = np.stack(images)
    actions = np.array(actions, dtype=np.float32)
    
    # Shuffle the data
    shuffle_idx = np.random.permutation(num_frames)
    images = images[shuffle_idx]
    actions = actions[shuffle_idx]
    
    with h5py.File(output_path, 'w') as f:
        f.attrs['sim'] = True
        f.create_dataset('/observations/images/main', data=images)
        f.create_dataset('/action', data=actions)


def create_demo_data_temporal(output_path, num_frames=1000):
    """Creates demonstration data with cyclic temporal pattern.
    
    Pattern: 1->2->3->4->5->6->7->8->9->8->7->6->5->4->3->2->1->2...
    
    Example of 16 consecutive frames:
    Frame idx:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
    Target:     1  2  3  4  5  6  7  8  9  8  7  6  5  4  3  2
    
    The qpos provides additional state information:
    qpos[i] = [sin(θ), cos(θ)] where θ represents position in cycle
    """
    images = np.random.randint(0, 255, (num_frames, 240, 240, 3), dtype=np.uint8)
    actions = []
    qpos = np.zeros((num_frames, 2))
    
    cycle_length = 9
    for i in range(num_frames):
        # Position in cycle (0 to 15 for complete up-down cycle)
        cycle_pos = i % (2 * cycle_length - 2)
        
        # Generate target value (1->9->1 pattern)
        if cycle_pos < cycle_length:
            target = cycle_pos + 1  # Going up: 1 to 9
        else:
            target = 2 * cycle_length - cycle_pos - 1  # Going down: 9 to 1
        
        actions.append([float(target)])
        
        # Generate state representation
        theta = 2 * np.pi * cycle_pos / (2 * cycle_length - 2)
        qpos[i] = [np.sin(theta), np.cos(theta)]
    
    actions = np.array(actions, dtype=np.float32)
    
    with h5py.File(output_path, 'w') as f:
        f.attrs['sim'] = True
        f.create_dataset('/observations/images/main', data=images)
        f.create_dataset('/observations/qpos', data=qpos)
        f.create_dataset('/action', data=actions)


def train(policy, train_loader, num_epochs=50, lr=1e-4):
    """Trains the policy network.
    
    DataLoader provides batches of samples:
    - With qpos: (images, qpos, targets) shapes: 
        images: (batch_size, 3, 240, 240)
        qpos: (batch_size, 2)
        targets: (batch_size,)
        
    - Without qpos: (images, targets) shapes:
        images: (batch_size, 3, 240, 240)
        targets: (batch_size,)
    
    The samples shown during training are the first sample from 
    10 consecutive batches, to show a sequence of predictions.
    """
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Get samples for visualization (first sample from several batches)
    samples = []
    for batch in train_loader:
        if len(samples) < 10:
            if len(batch) == 3:  # Dataset with qpos
                img, qpos, target = batch
                samples.append((img[0], qpos[0], target[0]))  # Take first sample from batch
            else:  # Dataset without qpos
                img, target = batch
                samples.append((img[0], None, target[0]))  # Take first sample from batch
        else:
            break
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            if len(batch) == 3:  # Dataset with qpos
                images, qpos, targets = batch
                images, qpos, targets = images.cpu(), qpos.cpu(), targets.cpu()
                outputs = policy(images, qpos)
            else:  # Dataset without qpos
                images, targets = batch
                images, targets = images.cpu(), targets.cpu()
                outputs = policy(images)
            
            loss = criterion(outputs.squeeze(), targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Show predictions on sample sequence
        with torch.no_grad():
            print(f'Epoch {epoch}:')
            print(f'  Average Loss: {total_loss/len(train_loader):.4f}')
            print("  Sample sequence predictions vs targets:")
            for i, (img, qpos, target) in enumerate(samples):
                if qpos is not None:
                    pred = policy(img.unsqueeze(0).cpu(), qpos.unsqueeze(0).cpu()).item()
                else:
                    pred = policy(img.unsqueeze(0).cpu()).item()
                print(f"    t{i}: {pred:.1f} vs {target:.1f}")
            print()


def main():
    data_path = 'demo_data.hdf5'
    if os.path.exists(data_path):
        os.remove(data_path)
    
    # Choose which dataset to create
    dataset_type = "temporal"  # "fixed", "black_white", or "temporal"
    
    if dataset_type == "fixed":
        create_demo_data_fixed(data_path, target_value=5.0)
    elif dataset_type == "black_white":
        create_demo_data_black_white(data_path)
    elif dataset_type == "temporal":
        create_demo_data_temporal(data_path)
    
    dataset = DemoDataset(data_path)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Check if dataset includes qpos
    sample = next(iter(train_loader))
    use_qpos = len(sample) == 3
    qpos_dim = sample[1].shape[1] if use_qpos else 0
    
    policy = SimplePolicy(image_size=240, use_qpos=use_qpos, qpos_dim=qpos_dim).cpu()
    train(policy, train_loader)
    
    torch.save(policy.state_dict(), 'simple_policy.pth')


if __name__ == '__main__':
    main() 