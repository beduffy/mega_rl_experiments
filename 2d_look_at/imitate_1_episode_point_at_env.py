import os
from datetime import datetime

import torch
import numpy as np
import h5py
import argparse
from torch import nn
from torch.utils.data import Dataset, DataLoader


class SimplePolicy(nn.Module):
    """Neural network that maps servo environment observations to actions.

    Input shapes:
        images: (batch_size, 3, 240, 240) - RGB images normalized to [0,1]
        qpos: (batch_size, 2) - Servo state [current_angle, velocity]

    Output shape:
        predictions: (batch_size, 1) - Predicted target angle in radians
    """

    def __init__(self, image_size=240, use_qpos=True, qpos_dim=2):
        super().__init__()

        # Calculate conv output size
        def conv2d_output_size(size, kernel_size=3, stride=2):
            return ((size - kernel_size) // stride) + 1

        size1 = conv2d_output_size(image_size)  # 119
        size2 = conv2d_output_size(size1)  # 59
        flat_size = 32 * size2 * size2  # 32 channels * 59 * 59

        # CNN processes images: (batch_size, 3, 240, 240) -> (batch_size, flat_size)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),  # (batch_size, 16, 119, 119)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (batch_size, 32, 59, 59)
            nn.ReLU(),
            nn.Flatten(),  # (batch_size, 32 * 59 * 59)
        )

        # MLP combines flattened image features with qpos
        combined_input_size = flat_size + qpos_dim
        self.mlp = nn.Sequential(
            nn.Linear(combined_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # Final output: predicted target angle
        )

        self.use_qpos = use_qpos

    def forward(self, images, qpos=None):
        x = self.conv(images)
        if self.use_qpos:
            x = torch.cat([x, qpos], dim=1)
        return self.mlp(x)


class ServoDataset(Dataset):
    """Dataset for servo environment demonstrations.

    Data structure in HDF5 file:
        /observations/
            images/main: (num_frames, 240, 240, 3) - RGB images
            qpos: (num_frames, 2) - Servo state [current_angle, velocity]
        /action: (num_frames, 1) - Target angles in radians
    """

    def __init__(self, data_path):
        with h5py.File(data_path, "r") as f:
            self.images = (
                torch.from_numpy(f["/observations/images/main"][:]).float() / 255.0
            )
            self.qpos = torch.from_numpy(f["/observations/qpos"][:]).float()
            self.actions = torch.from_numpy(f["/action"][:]).float()

        # Rearrange image dimensions from (N,H,W,C) to (N,C,H,W) for PyTorch
        self.images = self.images.permute(0, 3, 1, 2)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.qpos[idx], self.actions[idx]


def train(policy, train_loader, num_epochs=50, lr=1e-4, device="cpu"):
    """Trains the policy network on servo demonstrations."""
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
    os.makedirs("checkpoints", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (images, qpos, targets) in enumerate(train_loader):
            images = images.to(device)
            qpos = qpos.to(device)
            targets = targets.to(device)

            outputs = policy(images, qpos)
            loss = criterion(outputs.squeeze(), targets.squeeze())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Show predictions on sample sequence
        with torch.no_grad():
            print(f"Epoch {epoch}:")
            print(f"  Average Loss: {total_loss/len(train_loader):.4f}")
            print("  Sample predictions vs targets (in radians):")
            for i, (img, qpos, target) in enumerate(samples):
                img = img.unsqueeze(0).to(device)
                qpos = qpos.unsqueeze(0).to(device)
                pred = policy(img, qpos).item()
                print(f"    t{i}: {pred:.3f} vs {target.item():.3f}")
            print()

        # Save checkpoint every 100 epochs
        if (epoch + 1) % 100 == 0:
            checkpoint_path = f"checkpoints/servo_policy_{timestamp}_ep{epoch+1}.pth"
            torch.save(policy.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="scripted_episode.hdf5",
        help="Path to demonstration data",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=500, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to train on (cpu or cuda)",
    )
    args = parser.parse_args()

    # Set device
    device = torch.device(args.device)
    print(f"Training on {device}")

    # Load demonstration data
    dataset = ServoDataset(args.data_path)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Create and train policy
    policy = SimplePolicy(image_size=240, use_qpos=True, qpos_dim=2).to(device)
    train(policy, train_loader, num_epochs=args.num_epochs, lr=args.lr, device=device)

    # Save trained policy with timestamp and epochs
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"servo_policy_{timestamp}_ep{args.num_epochs}.pth"
    torch.save(policy.state_dict(), save_path)


if __name__ == "__main__":
    main()
