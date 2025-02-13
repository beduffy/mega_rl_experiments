import time
import numpy as np
from simulated_pixel_servo_point_flag_at_target import ServoEnv


def test_fps(render_mode, num_steps=1000):
    env = ServoEnv(render_mode=render_mode)
    env.reset()

    start_time = time.time()

    for _ in range(num_steps):
        action = np.random.uniform(-np.pi, np.pi)
        _, _, done, _ = env.step(action)
        if done:
            env.reset()

    end_time = time.time()
    elapsed = end_time - start_time
    fps = num_steps / elapsed

    return fps


def main():
    # Test with rendering
    print("Testing with rendering...")
    fps_rendered = test_fps(render_mode="human")
    print(f"FPS with rendering: {fps_rendered:.1f}")

    # Test headless
    print("\nTesting headless mode...")
    fps_headless = test_fps(render_mode=None)
    print(f"FPS in headless mode: {fps_headless:.1f}")

    # Print speedup
    speedup = fps_headless / fps_rendered
    print(f"\nHeadless mode is {speedup:.1f}x faster")


if __name__ == "__main__":
    main()
