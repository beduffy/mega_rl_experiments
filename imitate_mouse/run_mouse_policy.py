import time
import argparse

import numpy as np
import torch
import pyautogui

from imitate_mouse import MouseRecorder, ACTPolicy


def run_policy_eval(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    # Hardcoded policy config matching training parameters
    policy_config = {
        "num_queries": 100,  # Should match --chunk_size from training
        "hidden_dim": 512,
        "dim_feedforward": 3200,
        "lr_backbone": 1e-5,
        "backbone": "resnet18",
        "enc_layers": 4,
        "dec_layers": 7,
        "nheads": 8,
        "camera_names": ["mouse_cam"],
        "kl_weight": 10,
        "num_actions": 2,
        "state_dim": 2,
    }

    policy = ACTPolicy(policy_config).to(device)
    policy.load_state_dict(torch.load(args.ckpt))
    policy.eval()

    recorder = MouseRecorder()
    recorder.start_recording()

    try:
        while True:
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
