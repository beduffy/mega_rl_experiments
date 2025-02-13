import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
import argparse
from simulated_pixel_servo_point_flag_at_target import ServoEnv
import os
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Define experience tuple structure
Experience = namedtuple(
    "Experience", ["state", "action", "reward", "next_state", "done"]
)


# TODO check how much easier is it to learn local policy of left vs right, or velocity compared to absolute angle, surely absolute angle is the hardest


class DQN(nn.Module):
    """DQN that takes in images and qpos and outputs Q-values for discretized actions"""

    def __init__(self, image_size=240, qpos_dim=2, num_actions=16):
        super().__init__()

        # Calculate conv output size
        def conv2d_output_size(size, kernel_size=3, stride=2):
            return ((size - kernel_size) // stride) + 1

        size1 = conv2d_output_size(image_size)  # 119
        size2 = conv2d_output_size(size1)  # 59
        flat_size = 32 * size2 * size2  # 32 channels * 59 * 59

        # CNN for processing images
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # MLP for combining image features with qpos and outputting Q-values
        self.mlp = nn.Sequential(
            nn.Linear(flat_size + qpos_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),  # Output Q-value for each discrete action
        )

    def forward(self, image, qpos):
        x = self.conv(image)
        x = torch.cat([x, qpos], dim=1)
        return self.mlp(x)


class ReplayBuffer:
    """Experience replay buffer"""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def process_observation(obs):
    """Convert environment observation to torch tensors"""
    image, qpos = obs
    image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    qpos_tensor = torch.from_numpy(qpos).float().unsqueeze(0)
    return image_tensor, qpos_tensor


def train_step(policy_net, target_net, optimizer, experiences, device, gamma):
    """Perform one training step with a batch of experiences"""
    # Unpack experiences
    batch = Experience(*zip(*experiences))

    # Process states
    state_images = torch.cat([process_observation(s)[0] for s in batch.state]).to(
        device
    )
    state_qpos = torch.cat([process_observation(s)[1] for s in batch.state]).to(device)
    next_state_images = torch.cat(
        [process_observation(s)[0] for s in batch.next_state]
    ).to(device)
    next_state_qpos = torch.cat(
        [process_observation(s)[1] for s in batch.next_state]
    ).to(device)

    actions = torch.tensor(batch.action, device=device)
    rewards = torch.tensor(batch.reward, device=device)
    dones = torch.tensor(batch.done, dtype=torch.float, device=device)

    # Compute current Q values
    current_q_values = policy_net(state_images, state_qpos).gather(
        1, actions.unsqueeze(1)
    )

    # Compute next Q values using target network
    with torch.no_grad():
        next_q_values = target_net(next_state_images, next_state_qpos).max(1)[0]
        next_q_values[dones == 1] = 0  # Set to 0 for terminal states
        target_q_values = rewards + gamma * next_q_values

    # Compute loss and update
    loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate(env, policy_net, device, num_episodes=10, render=False):
    """Evaluate current policy"""
    rewards = []

    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0

        if render:
            env.start_recording()

        while True:
            image_tensor, qpos_tensor = process_observation(obs)

            with torch.no_grad():
                q_values = policy_net(image_tensor.to(device), qpos_tensor.to(device))
                action_idx = q_values.max(1)[1].item()

            action = (action_idx / 8.0 - 1.0) * np.pi
            obs, reward, done, _ = env.step(action)
            episode_reward += reward

            if done:
                break

        if render:
            env.stop_recording()
            env.save_recording(f"eval_episode_{episode}.hdf5")

        rewards.append(episode_reward)

    return np.mean(rewards), np.std(rewards)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.01)
    parser.add_argument("--epsilon_decay", type=float, default=0.995)
    parser.add_argument("--target_update", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_episodes", type=int, default=2)
    parser.add_argument("--save_dir", type=str, default="rl_servo_training")
    args = parser.parse_args()

    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "eval_videos"), exist_ok=True)

    # Initialize tensorboard
    writer = SummaryWriter(save_dir)

    # Initialize environment and networks
    env = ServoEnv()
    eval_env = ServoEnv()  # Separate env for evaluation
    device = torch.device(args.device)

    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)

    replay_buffer = ReplayBuffer()

    # Training metrics
    epsilon = args.epsilon_start
    best_eval_reward = float("-inf")
    episode_rewards = []
    losses = []

    for episode in range(args.num_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_loss = 0
        num_steps = 0

        while True:
            image_tensor, qpos_tensor = process_observation(obs)

            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action_idx = random.randrange(16)
            else:
                with torch.no_grad():
                    q_values = policy_net(
                        image_tensor.to(device), qpos_tensor.to(device)
                    )
                    action_idx = q_values.max(1)[1].item()

            action = (action_idx / 8.0 - 1.0) * np.pi
            next_obs, reward, done, _ = env.step(action)

            replay_buffer.push(obs, action_idx, reward, next_obs, done)

            if len(replay_buffer) >= args.batch_size:
                experiences = replay_buffer.sample(args.batch_size)
                loss = train_step(
                    policy_net, target_net, optimizer, experiences, device, args.gamma
                )
                episode_loss += loss

            obs = next_obs
            episode_reward += reward
            num_steps += 1

            if done:
                break

        # Update target network
        if episode % args.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Decay epsilon
        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)

        # Log metrics
        episode_rewards.append(episode_reward)
        if episode_loss > 0:
            losses.append(episode_loss / num_steps)

        writer.add_scalar("Train/Reward", episode_reward, episode)
        writer.add_scalar(
            "Train/Loss", episode_loss / num_steps if episode_loss > 0 else 0, episode
        )
        writer.add_scalar("Train/Epsilon", epsilon, episode)

        # Evaluate
        if episode % args.eval_interval == 0:
            mean_reward, std_reward = evaluate(
                eval_env,
                policy_net,
                device,
                num_episodes=args.eval_episodes,
                render=(
                    episode % (args.eval_interval * 5) == 0
                ),  # Record video every 5 evals
            )
            writer.add_scalar("Eval/Mean_Reward", mean_reward, episode)
            writer.add_scalar("Eval/Std_Reward", std_reward, episode)

            # Save best model
            if mean_reward > best_eval_reward:
                best_eval_reward = mean_reward
                torch.save(
                    {
                        "episode": episode,
                        "model_state_dict": policy_net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "reward": mean_reward,
                    },
                    os.path.join(save_dir, "checkpoints", "best_model.pth"),
                )

            print(
                f"Episode {episode}: Train reward = {episode_reward:.2f}, "
                f"Eval reward = {mean_reward:.2f} ± {std_reward:.2f}, "
                f"Epsilon = {epsilon:.2f}"
            )

        # Save checkpoint
        if episode % 100 == 0:
            torch.save(
                {
                    "episode": episode,
                    "model_state_dict": policy_net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "reward": episode_reward,
                },
                os.path.join(save_dir, "checkpoints", f"checkpoint_ep{episode}.pth"),
            )

    # Plot final training curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"))
    plt.close()


if __name__ == "__main__":
    main()
