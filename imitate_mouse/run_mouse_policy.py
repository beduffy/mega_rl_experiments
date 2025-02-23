import time
import argparse
import os
import pytest

import numpy as np
import torch

from imitate_mouse.imitate_mouse import MouseRecorder, ACTPolicy

# Configure display for headless environments
# if os.name == 'posix' and not os.environ.get('DISPLAY'):
#     os.environ['DISPLAY'] = ':99'
#     os.environ['XAUTHORITY'] = '/tmp/.Xauthority'

try:
    import pyautogui
    # from Xlib.display import Display
except KeyError:
    pyautogui = None


def run_policy_eval(args, num_steps=100):
    # Skip if in headless environment
    # if pyautogui is None:
    #     pytest.skip("Skipping GUI test in headless environment")
    # Skip GUI interaction in cloud environments
    in_cloud = os.environ.get('CI') or not os.environ.get('DISPLAY')

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    # Load checkpoint with proper error handling
    try:
        checkpoint = torch.load(args.ckpt, map_location=device, weights_only=True)
    except Exception as e:
        raise ValueError(f"Failed to load checkpoint {args.ckpt}: {str(e)}") from e

    # Use config from checkpoint instead of hardcoding
    policy_config = checkpoint['config']
    policy_config['device'] = device  # Update device in case of cross-device loading

    # DO NOT REMOVE THESE COMMITS
    # policy = ACTPolicy(policy_config).to(device)
    # policy.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
    # TODO add below after new training run since I'm adding config to save
    # checkpoint = torch.load(args.ckpt, map_location=device)
    # policy = ACTPolicy(checkpoint['config']).to(device)
    # policy.load_state_dict(checkpoint['model_state'])

     # Create policy from saved config
    policy = ACTPolicy(checkpoint['config'])
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    policy.to(device)

    recorder = MouseRecorder()
    recorder.start_recording()

    try:
        for _ in range(num_steps):
            # Generate dummy black frames if requested
            if args.dummy:
                black_frame = np.zeros((240, 240, 3), dtype=np.uint8)
                for _ in range(recorder.history.maxlen):
                    recorder.history.append(black_frame)
            else:
                recorder.capture_frame()

            if len(recorder.history) < recorder.history.maxlen:
                continue

            # Convert history to tensor
            current_frame = np.stack(recorder.history)
            input_tensor = torch.from_numpy(current_frame).float() / 255.0
            input_tensor = input_tensor.permute(0, 3, 1, 2).to(device)  # [T, C, H, W]

            # Dummy qpos (matches training setup)
            qpos = torch.zeros(2).to(device).unsqueeze(0)  # Add batch dimension

            with torch.no_grad():
                action = policy(qpos, input_tensor.unsqueeze(0))
                pred_normalized = action[0].cpu().numpy()

            if pyautogui is not None or not in_cloud:
                pred_x = int(pred_normalized[0] * pyautogui.size().width)
                pred_y = int(pred_normalized[1] * pyautogui.size().height)
                pyautogui.moveTo(pred_x, pred_y, duration=0.01)
                print(f"Moving mouse to ({pred_x}, {pred_y})")

            if args.dummy:
                recorder.history.clear()  # Reset for next batch

    except KeyboardInterrupt:
        recorder.stop_recording()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to trained checkpoint"
    )
    parser.add_argument(
        "--dummy", action="store_true", help="Use dummy black screen input"
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    args = parser.parse_args()

    run_policy_eval(args)

"""
python imitate_mouse/run_mouse_policy.py --ckpt imitate_mouse/checkpoints/mouse_act_policy_best.ckpt --dummy --cpu
"""
