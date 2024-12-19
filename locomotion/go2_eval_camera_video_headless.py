import argparse
import os
import pickle

import numpy as np
import torch
from go2_env import Go2Env
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("--ckpt", type=int, default=100)
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")

    # print(env.scene)
    # import pdb;pdb.set_trace()
    # cam = env.scene.add_camera(
    #     res    = (1280, 960),
    #     pos    = (3.5, 0.0, 2.5),
    #     lookat = (0, 0, 0.5),
    #     fov    = 30,x
    #     GUI    = False
    # )

    obs, _ = env.reset()
    env.cam.start_recording()
    with torch.no_grad():
        frame_count = 0
        while True:
            actions = policy(obs)
            obs, _, rews, dones, infos = env.step(actions)
            frame_count += 1
            # print('obs: ', obs)

            # change camera position
            env.cam.set_pose(
                pos    = (3.0 * np.sin(frame_count / 60), 3.0 * np.cos(frame_count / 60), 2.5),
                lookat = (0, 0, 0.5),
            )
            
            env.cam.render()

            if frame_count > 500:
                break

    # stop recording and save video. If `filename` is not specified, a name will be auto-generated using the caller file name.
    env.cam.stop_recording(save_to_filename='{}.mp4'.format(args.exp_name), fps=60)


if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go2_eval.py -e go2-walking -v --ckpt 100
"""
