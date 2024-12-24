import pybullet as p
import numpy as np
import time


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

        target_position = [0.0, 0.0, 0.0]
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_position,
            cameraTargetPosition=target_position,
            cameraUpVector=[0, 0, 1]
        )
        
        # Enable mouse picking and keyboard control
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 1)
        
        # Cache the last camera parameters
        self._last_yaw = None
        self._last_pitch = None
        self._last_position = None
        self._cached_view_matrix = None
        
        # Pre-allocate arrays for better performance
        self._target_position = np.zeros(3, dtype=np.float32)
        
        self._frame_counter = 0
        self._last_cleanup_time = time.time()
        
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
        self._frame_counter += 1
        current_time = time.time()
        
        # Force cleanup every 1000 frames or 30 seconds
        # if self._frame_counter >= 200 or (current_time - self._last_cleanup_time) > 30:
        #     print("Performing PyBullet cleanup...")
        #     # Store current state
        #     temp_yaw = self.yaw
        #     temp_pitch = self.pitch
        #     temp_pos = self.camera_position.copy()
            
        #     # Force PyBullet cleanup
        #     p.resetSimulation()
        #     p.resetDebugVisualizerCamera(
        #         cameraDistance=5,
        #         cameraYaw=0,
        #         cameraPitch=-20,
        #         cameraTargetPosition=[0, 0, 0]
        #     )
            
        #     # Restore state
        #     self.yaw = temp_yaw
        #     self.pitch = temp_pitch
        #     self.camera_position = temp_pos
        #     self._last_yaw = None  # Force view matrix recomputation
        #     self._frame_counter = 0
        #     self._last_cleanup_time = current_time
        
        # Only recompute view matrix if camera has moved
        if (self._last_yaw != self.yaw or 
            self._last_pitch != self.pitch or 
            not np.array_equal(self._last_position, self.camera_position)):
            
            # Update cached values
            self._last_yaw = self.yaw
            self._last_pitch = self.pitch
            self._last_position = np.array(self.camera_position, dtype=np.float32)
            
            # Calculate look-at point based on camera rotation
            yaw_rad = np.radians(self.yaw)
            pitch_rad = np.radians(self.pitch)
            
            # Calculate direction vector and store directly in target_position
            self._target_position[0] = self.camera_position[0] + np.cos(yaw_rad) * np.cos(pitch_rad)
            self._target_position[1] = self.camera_position[1] + np.sin(yaw_rad) * np.cos(pitch_rad)
            self._target_position[2] = self.camera_position[2] + np.sin(pitch_rad)
            
            # Compute and cache view matrix
            self._cached_view_matrix = p.computeViewMatrix(
                cameraEyePosition=self._last_position,
                cameraTargetPosition=self._target_position,
                cameraUpVector=[0, 0, 1]
            )
        
        # Use cached view matrix with minimal flags
        (_, _, px, _, _) = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self._cached_view_matrix,
            projectionMatrix=self.proj_matrix,
            renderer=p.ER_TINY_RENDERER,
            shadow=0
        )
        
        return np.frombuffer(px, dtype=np.uint8).reshape(self.height, self.width, 4)[:, :, :3]
    

    def print_current_camera_params(self):
        """Print current camera parameters"""
        params = p.getDebugVisualizerCamera()
        print("\nCamera Parameters:")
        print(f"Distance: {params[10]:.2f}")
        print(f"Yaw: {params[8]:.2f}")
        print(f"Pitch: {params[9]:.2f}")
        print(f"Target Position: {params[11]}")
