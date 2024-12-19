import time

import pybullet as p
import pybullet_data
import numpy as np

# from simulation_utils import CameraController


class CameraController:
    def __init__(self, distance=2.0, yaw=45.0, pitch=-30.0, target_position=[0.0, 0.0, 0.0]):
        # Default camera parameters
        self.distance = distance
        self.yaw = yaw
        self.pitch = pitch
        self.target_position = target_position
        self.up_axis_index = 2  # Z-axis up (0=X, 1=Y, 2=Z)
        
        # Enable mouse picking and keyboard control
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 1)
        
        # Initialize camera
        self.reset_camera()
    

    def reset_camera(self):
        """Reset camera to default position"""
        p.resetDebugVisualizerCamera(
            cameraDistance=self.distance,
            cameraYaw=self.yaw,
            cameraPitch=self.pitch,
            cameraTargetPosition=self.target_position
        )
    

    def set_camera_position(self, distance, yaw, pitch, target_position=None):
        """Programmatically set camera position"""
        self.distance = distance
        self.yaw = yaw
        self.pitch = pitch
        if target_position is not None:
            self.target_position = target_position
        
        p.resetDebugVisualizerCamera(
            cameraDistance=self.distance,
            cameraYaw=self.yaw,
            cameraPitch=self.pitch,
            cameraTargetPosition=self.target_position
        )
    

    def orbit_camera(self, orbit_speed=0.5):
        """Orbit camera around target position"""
        self.yaw += orbit_speed
        if self.yaw > 360:
            self.yaw -= 360
        
        p.resetDebugVisualizerCamera(
            cameraDistance=self.distance,
            cameraYaw=self.yaw,
            cameraPitch=self.pitch,
            cameraTargetPosition=self.target_position
        )
    

    def get_camera_image(self, width=640, height=480):
        """Get RGB and depth image from current camera position"""
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.target_position,
            distance=self.distance,
            yaw=self.yaw,
            pitch=self.pitch,
            roll=0,
            upAxisIndex=self.up_axis_index
        )
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60.0,
            aspect=width / height,
            nearVal=0.1,
            farVal=100.0
        )
        
        # Get camera image
        (_, _, px, depth, _) = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = rgb_array[:, :, :3]
        
        return rgb_array, depth
    

    def print_current_camera_params(self):
        """Print current camera parameters"""
        params = p.getDebugVisualizerCamera()
        print("\nCamera Parameters:")
        print(f"Distance: {params[10]:.2f}")
        print(f"Yaw: {params[8]:.2f}")
        print(f"Pitch: {params[9]:.2f}")
        print(f"Target Position: {params[11]}")


class LookAtObjectEnv:
    def __init__(self):
        # Connect to PyBullet
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Initialize camera controller
        self.camera = CameraController()
        
        # Create red sphere target
        self.sphere_pos = [0.0, 0.0, 1.0]
        sphere_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1)
        sphere_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 0, 0, 1])
        self.sphere_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=sphere_collision,
            baseVisualShapeIndex=sphere_visual,
            basePosition=self.sphere_pos
        )
    

    def get_observation(self):
        """Get RGB image and calculate distance to sphere center"""
        rgb, depth = self.camera.get_camera_image()
        
        # Find red sphere in image (simple color thresholding)
        red_mask = (rgb[:,:,0] > 200) & (rgb[:,:,1] < 50) & (rgb[:,:,2] < 50)
        sphere_pixels = np.where(red_mask)
        
        if len(sphere_pixels[0]) == 0:
            # Sphere not in view
            return rgb, None
        
        # Calculate sphere center in image
        sphere_center_y = np.mean(sphere_pixels[0])
        sphere_center_x = np.mean(sphere_pixels[1])
        
        # Calculate image center
        image_center_y = rgb.shape[0] / 2
        image_center_x = rgb.shape[1] / 2
        
        # Calculate Euclidean distance from image center to sphere center
        distance = np.sqrt((sphere_center_x - image_center_x)**2 + 
                         (sphere_center_y - image_center_y)**2)
        
        return rgb, distance
    

    def step(self, action):
        """Take an action to move the camera
        action: dict with possible keys 'yaw', 'pitch', 'distance'
        """
        if 'yaw' in action:
            self.camera.yaw += action['yaw']
        if 'pitch' in action:
            self.camera.pitch += action['pitch']
        if 'distance' in action:
            self.camera.distance += action['distance']
            
        self.camera.set_camera_position(
            self.camera.distance,
            self.camera.yaw,
            self.camera.pitch
        )
        
        rgb, distance = self.get_observation()
        return rgb, distance
    

    def reset(self):
        """Reset camera to default position"""
        self.camera.reset_camera()

        return self.get_observation()


def main():
    env = LookAtObjectEnv()
    
    print("\nCamera Controls:")
    print("- Mouse left button: Rotate camera")
    print("- Mouse right button: Zoom camera")
    print("- Mouse wheel: Zoom camera")
    print("- Mouse middle button + drag: Pan camera")
    print("- Press 'r' to reset camera")
    print("- Press 'p' to print current camera parameters")
    print("- Press 'q' to quit")
    
    yaw = 0
    pitch = 0
    rgb, distance = env.get_observation()

    while True:
        # Handle keyboard input
        keys = p.getKeyboardEvents()
        
        for key, state in keys.items():
            if state == 3:  # Key press event
                if key == ord('r'):
                    rgb, distance = env.reset()
                    if distance is not None:
                        print(f"Distance to sphere center: {distance:.2f}")
                elif key == ord('p'):
                    env.camera.print_current_camera_params()
                elif key == ord('q'):
                    p.disconnect()
                    return
                elif key == ord('a'):
                    env.camera.yaw -= 5
                    env.camera.set_camera_position(
                        env.camera.distance,
                        env.camera.yaw,
                        env.camera.pitch
                    )
                    rgb, distance = env.get_observation()
                    if distance is not None:
                        print(f"Distance to sphere center: {distance:.2f}")
                elif key == ord('d'):
                    env.camera.yaw += 5
                    env.camera.set_camera_position(
                        env.camera.distance,
                        env.camera.yaw,
                        env.camera.pitch
                    )
                    rgb, distance = env.get_observation()
                    if distance is not None:
                        print(f"Distance to sphere center: {distance:.2f}")
                elif key == ord('w'):
                    env.camera.pitch += 5
                    env.camera.set_camera_position(
                        env.camera.distance,
                        env.camera.yaw,
                        env.camera.pitch
                    )
                    rgb, distance = env.get_observation()
                    if distance is not None:
                        print(f"Distance to sphere center: {distance:.2f}")
                elif key == ord('s'):
                    env.camera.pitch -= 5
                    env.camera.set_camera_position(
                        env.camera.distance,
                        env.camera.yaw,
                        env.camera.pitch
                    )
                    rgb, distance = env.get_observation()
                    if distance is not None:
                        print(f"Distance to sphere center: {distance:.2f}")
        
        # Example: Take random actions
        # if int(time.time()) % 3 == 0:
        
            # action = {
            #     'yaw': np.random.uniform(-5, 5),
            #     'pitch': np.random.uniform(-5, 5)
            # }
            # rgb, distance = env.step(action)
        if distance is not None:
            print(f"Distance to sphere center: {distance:.2f}")
        
        p.stepSimulation()
        time.sleep(1./240.)


if __name__ == "__main__":
    main()