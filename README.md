# Robotics Learning Projects

![Python Tests](https://github.com/[owner]/[repository]/actions/workflows/test.yml/badge.svg)


A collection of reinforcement learning and imitation learning projects for robotic control tasks.

## Project Structure

### 1. 2D Look-At Task (Visual Servoing)

- Imitation learning for 2D visual servoing
- Contains dataset handling and simple CNN policy
- Includes synthetic data generation utilities

- Soft Actor-Critic implementation for servo control
- Handles both image and proprioceptive (qpos) inputs
- GPU-optimized replay buffer implementation

### 2. Quadruped Locomotion (Unitree Go2)

- Full physics environment for Unitree Go2 robot
- Implements reward functions and PD control
- Supports parallel environments for efficient training

### 3. 3D Look-At Object Task

- PyBullet-based environment for object fixation task
- Discrete action space for camera control
- Includes memory debugging utilities

### 4. Imitation Learning Projects

- Sequence prediction policy for multi-step actions
- Temporal convolutional network architecture
- PyBullet deployment integration

- Mouse movement imitation from screen capture
- Temporal CNN with spatial pyramid pooling
- Normalized coordinate prediction

## Installation

```bash
pip install -r requirements.txt
# Core dependencies:
# - PyTorch
# - PyBullet
# - h5py
# - stable-baselines3
# - genesis-sim
```