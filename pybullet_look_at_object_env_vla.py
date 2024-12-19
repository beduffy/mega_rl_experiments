import pybullet as p
import pybullet_data
import time
import numpy as np
from simulation_utils import CameraController


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