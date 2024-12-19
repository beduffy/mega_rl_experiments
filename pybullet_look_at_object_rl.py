import time
import enum
from typing import Tuple, Optional

import pybullet as p
import pybullet_data
import numpy as np

from py

def human_control_main():
    """Test environment with keyboard controls"""
    env = LookAtObjectEnv()
    
    print("\nCamera Controls:")
    print("- A/D: Rotate camera left/right")
    print("- W/S: Move forward/backward")
    print("- Q/E: Strafe left/right")
    print("- Up/Down: Look up/down")
    print("- R: Reset environment")
    print("- P: Print camera parameters")
    print("- Esc: Quit")
    
    print_interval = 0.1
    last_print_time = time.time()
    
    rgb, info = env.reset()
    
    try:
        while True:
            keys = p.getKeyboardEvents()
            
            for key, state in keys.items():
                if state & p.KEY_WAS_TRIGGERED or state & p.KEY_IS_DOWN:
                    action = None
                    
                    if key == ord('a'):
                        action = CameraAction.YAW_LEFT
                    elif key == ord('d'):
                        action = CameraAction.YAW_RIGHT
                    elif key == ord('w'):
                        action = CameraAction.MOVE_FORWARD
                    elif key == ord('s'):
                        action = CameraAction.MOVE_BACKWARD
                    elif key == ord('q'):
                        action = CameraAction.STRAFE_LEFT
                    elif key == ord('e'):
                        action = CameraAction.STRAFE_RIGHT
                    elif key == p.B3G_UP_ARROW:
                        action = CameraAction.PITCH_UP
                    elif key == p.B3G_DOWN_ARROW:
                        action = CameraAction.PITCH_DOWN
                    elif key == ord('r'):
                        rgb, info = env.reset()
                    elif key == ord('p'):
                        env.camera.print_current_camera_params()
                    
                    if action is not None:
                        rgb, reward, terminated, truncated, info = env.step(action)
                        
                        if time.time() - last_print_time >= print_interval:
                            distance_str = f"{info['distance']:.2f}" if info['distance'] is not None else "None"
                            print(f"Step {env.current_step}: reward={reward:.2f}, distance={distance_str}")
                            last_print_time = time.time()
                        
                        if terminated or truncated:
                            print("Episode finished!")
                            rgb, info = env.reset()
            
            p.stepSimulation()
            time.sleep(1./240.)
    
    finally:
        env.close()


if __name__ == "__main__":
    human_control_main()