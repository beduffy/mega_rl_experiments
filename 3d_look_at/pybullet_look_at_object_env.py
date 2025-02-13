import time
import enum
from typing import Tuple, Optional

import pybullet as p
import pybullet_data
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from camera_controller import CameraController
from memory_debug import memory_tracker, profile

# TODO do I want action primitives or some form of continuous action e.g. directly set yaw to 85 degrees rather than lots of discrete actions
# TODO claude 3.5 did badly. I think I need a real VLA model.
# TODO does it get slower over time?
# TODO curriculum learning, first begin with left vs right,, then larger randomness in left vs right, then anywhere so it has to hunt for it


class CameraAction(enum.IntEnum):
    """Discrete actions for camera control"""

    YAW_LEFT = 0
    YAW_RIGHT = 1
    MOVE_FORWARD = 2
    MOVE_BACKWARD = 3
    # TODO removing some for now to make problem easier.
    # TODO could do experiments with many actions that do nothing. credit assignment lol
    # PITCH_UP = 2
    # PITCH_DOWN = 3
    # STRAFE_LEFT = 6
    # STRAFE_RIGHT = 7
    # NO_OP = 8


class LookAtObjectEnv(gym.Env):
    """Environment for training an agent to look at a target object"""

    def __init__(self, render_mode="human"):
        super().__init__()
        self.render_mode = render_mode
        self.connection_mode = p.GUI if render_mode == "human" else p.DIRECT
        p.connect(self.connection_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Reduce image size for faster processing
        self.image_width = 128  # Reduced from 640
        self.image_height = 96  # Reduced from 480

        # Define action and observation spaces with new image size
        self.action_space = spaces.Discrete(len(CameraAction))
        print(f"Action space: {self.action_space}")
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.image_height, self.image_width, 3),
            dtype=np.uint8,
        )

        # Movement parameters
        self.yaw_delta = 2.0
        self.pitch_delta = 2.0
        self.translation_delta = 0.1

        # Reward parameters
        self.max_steps = 200
        self.current_step = 0
        self.target_distance_threshold = 10.0
        self.previous_distance = None

        # Target position
        self.target_position = [0.0, 0.0, 2.0]

    def close(self):
        """Clean up environment resources"""
        if p.isConnected():
            p.disconnect()

    # @profile
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """Reset the environment"""
        # Disconnect previous session if exists
        # if p.isConnected():
        #     p.disconnect()

        # Connect to PyBullet
        # # p.connect(self.connection_mode)
        # p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Initialize the RNG
        # super().reset(seed=seed)

        p.resetSimulation()

        # Load ground plane for reference
        p.loadURDF("plane.urdf")

        # Reset camera
        if not hasattr(self, "camera"):
            self.camera = CameraController()
        else:
            self.camera.reset()

        # Calculate sphere position based on camera's initial view
        # yaw_rad = np.radians(self.camera.yaw)
        # pitch_rad = np.radians(self.camera.pitch)

        # TODO camera controller class is completely messed up in terms of params of direction etc.

        # Position sphere 3 units ahead of camera in the direction it's looking
        # distance_ahead = 3.0
        # self.sphere_pos = [
        #     self.camera.camera_position[0] + distance_ahead * np.cos(yaw_rad) * np.cos(pitch_rad),
        #     self.camera.camera_position[1] + distance_ahead * np.sin(yaw_rad) * np.cos(pitch_rad),
        #     self.camera.camera_position[2] + distance_ahead * np.sin(pitch_rad)
        # ]

        # self.sphere_pos = [1, 3.0, 2.0]
        # self.sphere_pos = [0.3420201539993286, -2.060307264328003, 2.0]
        # self.sphere_pos = [3.0, 2.5, 2.0]  # finally, this is to the left if camera starts origin
        # self.sphere_pos = [3.0, -2.5, 2.0]  # finally, this is to the right
        # self.sphere_pos[1] = np.random.choice([0.2, -0.05])
        # self.sphere_pos[1] = np.random.choice([-0.2])
        # Randomize x position between -5 and 5
        # self.sphere_pos[0] = np.random.uniform(-5.0, 5.0)
        # self.sphere_pos[0] = -0.05

        # left vs right
        positions = [(3.0, 2.5, 2.0), (3.0, -2.5, 2.0)]
        self.sphere_pos = positions[np.random.choice(2)]

        # Create or reset target sphere
        radius = 0.2
        # if not hasattr(self, 'sphere_id'):
        sphere_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        sphere_visual = p.createVisualShape(
            p.GEOM_SPHERE, radius=radius, rgbaColor=[1, 0, 0, 1]
        )
        self.sphere_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=sphere_collision,
            baseVisualShapeIndex=sphere_visual,
            basePosition=self.sphere_pos,
        )
        # else:
        #     p.resetBasePositionAndOrientation(self.sphere_id, self.sphere_pos, [0, 0, 0, 1])

        # Reset episode variables
        self.current_step = 0
        rgb, distance = self.get_observation()
        self.previous_distance = distance

        # Optimize PyBullet settings
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        # p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
        # p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

        # p.setTimeStep(1./60.)  # Reduce simulation frequency
        p.setRealTimeSimulation(0)  # Disable real-time simulation

        info = {"distance": distance}
        # memory_tracker.log_memory("End reset")
        return rgb, info

    # @profile
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment

        Args:
            action (int): Action from CameraAction enum

        Returns:
            observation (np.ndarray): RGB image
            reward (float): Reward for the action
            terminated (bool): Whether episode is done
            truncated (bool): Whether episode was truncated
            info (dict): Additional information
        """
        self.current_step += 1

        # Apply action
        if action == CameraAction.YAW_LEFT:
            self.camera.yaw += self.yaw_delta
        elif action == CameraAction.YAW_RIGHT:
            self.camera.yaw -= self.yaw_delta
        # elif action == CameraAction.PITCH_UP:
        #     self.camera.pitch = min(89, self.camera.pitch + self.pitch_delta)
        # elif action == CameraAction.PITCH_DOWN:
        #     self.camera.pitch = max(-89, self.camera.pitch - self.pitch_delta)
        elif action == CameraAction.MOVE_FORWARD:
            self.camera.move_camera(
                forward=self.translation_delta
            )  # This should move forward
        elif action == CameraAction.MOVE_BACKWARD:
            self.camera.move_camera(
                forward=-self.translation_delta
            )  # This should move backward
        # elif action == CameraAction.STRAFE_LEFT:
        #     self.camera.move_camera(right=-self.translation_delta)     # This should strafe left
        # elif action == CameraAction.STRAFE_RIGHT:
        #     self.camera.move_camera(right=self.translation_delta)      # This should strafe right

        self.camera.update_camera()

        # Get new observation
        rgb, distance = self.get_observation()

        # Calculate reward
        reward = self._calculate_reward(distance)

        # Check if done
        terminated = False
        if distance is not None and distance < self.target_distance_threshold:
            print("Target centered in view!!!!\n\n")
            terminated = True  # Success - target centered in view
            reward += 10.0  # Bonus reward for success

        truncated = self.current_step >= self.max_steps

        # Update previous distance
        self.previous_distance = distance

        info = {
            "distance": distance,
            "step": self.current_step,
        }

        # memory_tracker.log_memory("End step")
        return rgb, reward, terminated, truncated, info

    def _calculate_reward(self, distance: Optional[float]) -> float:
        """Calculate reward based on current observation

        Args:
            distance (float): Distance from image center to target center

        Returns:
            float: Reward value
        """
        if distance is None:
            return -1.0  # Penalty for losing sight of target

        if self.previous_distance is None:
            return 0.0

        # Reward for moving closer to target
        reward = self.previous_distance - distance

        # Small step penalty to encourage efficiency
        reward -= 0.1

        return reward

    # @profile
    def get_observation(self) -> Tuple[np.ndarray, Optional[float]]:
        """Get current observation from environment"""
        start_time = time.time()

        # Get camera image (now without depth)
        self.rgb = self.camera.get_camera_image()  # Now returns just the rgb array

        # Fix red mask calculation - use & for element-wise AND
        red_mask = (
            (self.rgb[:, :, 0] > 200)
            & (self.rgb[:, :, 1] < 50)
            & (self.rgb[:, :, 2] < 50)
        )
        sphere_pixels = np.nonzero(red_mask)

        # Calculate distance using pre-computed image center
        if not hasattr(self, "_image_center"):
            self._image_center = np.array(
                [self.rgb.shape[0] / 2, self.rgb.shape[1] / 2]
            )
        # print('red_mask.shape:', red_mask.shape)
        # print('sphere_pixels:', sphere_pixels)
        if len(sphere_pixels[0]) == 0:
            self._last_rgb = self.rgb
            self._last_distance = None
            # return rgb, None
            distance = None

        else:
            # Calculate sphere center and distance
            sphere_center = np.array(
                [
                    np.mean(sphere_pixels[0]),  # y coordinate
                    np.mean(sphere_pixels[1]),  # x coordinate
                ]
            )

            distance = np.linalg.norm(sphere_center - self._image_center)

        # Cache results
        # self._last_rgb = rgb
        # self._last_distance = distance

        # Timing code
        # current_time = time.time()
        # if True:  # or use your preferred condition
        #     execution_time = current_time - start_time
        #     print(f"get_observation() took {execution_time*1000:.2f}ms")

        # memory_tracker.log_memory("End get_observation")
        return self.rgb, distance


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

    print("pybullet numpy enabled:", p.isNumpyEnabled())  # it is

    print_interval = 0.1
    last_print_time = time.time()

    rgb, info = env.reset()

    try:
        while True:
            keys = p.getKeyboardEvents()

            for key, state in keys.items():
                if state & p.KEY_WAS_TRIGGERED or state & p.KEY_IS_DOWN:
                    action = None

                    if key == ord("a"):
                        action = CameraAction.YAW_LEFT
                    elif key == ord("d"):
                        action = CameraAction.YAW_RIGHT
                    elif key == ord("w"):
                        action = CameraAction.MOVE_FORWARD
                    elif key == ord("s"):
                        action = CameraAction.STRAFE_RIGHT
                    elif key == ord("q"):
                        action = CameraAction.STRAFE_LEFT
                    elif key == ord("e"):
                        action = CameraAction.STRAFE_RIGHT
                    elif key == p.B3G_UP_ARROW:
                        action = CameraAction.PITCH_UP
                    elif key == p.B3G_DOWN_ARROW:
                        action = CameraAction.PITCH_DOWN
                    elif key == ord("r"):
                        rgb, info = env.reset()
                    elif key == ord("p"):
                        env.camera.print_current_camera_params()

                    if action is not None:
                        rgb, reward, terminated, truncated, info = env.step(action)

                        if time.time() - last_print_time >= print_interval:
                            distance_str = (
                                f"{info['distance']:.2f}"
                                if info["distance"] is not None
                                else "None"
                            )
                            print(
                                f"Step {env.current_step}: reward={reward:.2f}, distance={distance_str}"
                            )
                            last_print_time = time.time()

                        if terminated or truncated:
                            print("Episode finished!")
                            rgb, info = env.reset()

            p.stepSimulation()
            time.sleep(1.0 / 240.0)

    finally:
        env.close()


if __name__ == "__main__":
    human_control_main()
