import pybullet as p
import numpy as np


class CameraController:
    def __init__(self, distance=2.0, yaw=0.0, pitch=0.0, width=128, height=96):
        # Camera position in world space
        self.camera_position = [0.0, 0.0, 2.0]  # Starting position
        self.yaw = yaw
        self.pitch = pitch
        self.up_axis_index = 2
        self.initial_position = self.camera_position.copy()
        self.initial_yaw = yaw
        self.initial_pitch = pitch
        self.width = width
        self.height = height

        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=60.0,
            aspect=width / height,
            nearVal=0.1,
            farVal=100.0
        )
        
        # Enable mouse picking and keyboard control
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 1)
        
        self.update_camera()
    

    def reset(self):
        """Reset camera to initial position and orientation"""
        self.camera_position = self.initial_position.copy()
        self.yaw = self.initial_yaw
        self.pitch = self.initial_pitch
        self.update_camera()
    

    def update_camera(self):
        """Update camera view based on current position and rotation"""
        # Calculate look-at point based on camera rotation
        yaw_rad = np.radians(self.yaw)
        pitch_rad = np.radians(self.pitch)
        
        # Calculate direction vector
        look_x = np.cos(yaw_rad) * np.cos(pitch_rad)
        look_y = np.sin(yaw_rad) * np.cos(pitch_rad)
        look_z = np.sin(pitch_rad)
        
        # Calculate target position 1 unit ahead of camera
        target_position = [
            self.camera_position[0] + look_x,
            self.camera_position[1] + look_y,
            self.camera_position[2] + look_z
        ]
        
        # Update debug visualizer camera with a single call
        p.resetDebugVisualizerCamera(
            cameraDistance=0.1,
            cameraYaw=self.yaw - 90,
            cameraPitch=self.pitch,
            cameraTargetPosition=target_position
        )
    

    def move_camera(self, forward=0, right=0, up=0):
        """Move camera in its local coordinate system"""
        # Use yaw - 90 to match the visualization orientation
        yaw_rad = np.radians(self.yaw - 90)
        
        # Forward/backward movement - along the look direction
        self.camera_position[0] += -forward * np.sin(yaw_rad)
        self.camera_position[1] += forward * np.cos(yaw_rad)
        
        # Left/right movement - perpendicular to look direction
        self.camera_position[0] += right * np.cos(yaw_rad)
        self.camera_position[1] += right * np.sin(yaw_rad)
        
        # Up/down movement in world space
        self.camera_position[2] += up
        
        self.update_camera()
    

    def get_camera_image(self):
        """Get RGB and depth image from current camera position"""
        # Calculate look-at point based on camera rotation
        yaw_rad = np.radians(self.yaw)
        pitch_rad = np.radians(self.pitch)
        
        # Calculate direction vector
        look_x = np.cos(yaw_rad) * np.cos(pitch_rad)
        look_y = np.sin(yaw_rad) * np.cos(pitch_rad)
        look_z = np.sin(pitch_rad)
        
        # Calculate target position 1 unit ahead of camera
        target_position = [
            self.camera_position[0] + look_x,
            self.camera_position[1] + look_y,
            self.camera_position[2] + look_z
        ]
        
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_position,
            cameraTargetPosition=target_position,
            cameraUpVector=[0, 0, 1]
        )
        
        # Get camera image
        (_, _, px, depth, _) = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view_matrix,
            projectionMatrix=self.proj_matrix,
            renderer=p.ER_TINY_RENDERER,
            # flags=p.ER_NO_SEGMENTATION_MASK,  # Disable segmentation mask
            # shadow=0,  # Disable shadows
            # lightDirection=[0, 0, -1]  # Simple lighting
        )
        
        # rgb_array = np.array(px, dtype=np.uint8)
        # rgb_array = rgb_array[:, :, :3]
        # Optimize numpy operations
        # Avoid copy by using correct shape from the start
        rgb_array = np.frombuffer(px, dtype=np.uint8).reshape(self.height, self.width, 4)[:, :, :3]
        
        return rgb_array, depth
    

    def print_current_camera_params(self):
        """Print current camera parameters"""
        params = p.getDebugVisualizerCamera()
        print("\nCamera Parameters:")
        print(f"Distance: {params[10]:.2f}")
        print(f"Yaw: {params[8]:.2f}")
        print(f"Pitch: {params[9]:.2f}")
        print(f"Target Position: {params[11]}")
