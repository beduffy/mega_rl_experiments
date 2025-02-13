from memory_profiler import profile
import objgraph
import gc
import os
import psutil
import time


class MemoryTracker:
    def __init__(self, log_interval=1.0):
        self.process = psutil.Process(os.getpid())
        self.last_log_time = time.time()
        self.log_interval = log_interval
        self.start_rss = self.process.memory_info().rss / 1024 / 1024  # MB

    def log_memory(self, location=""):
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            # Force garbage collection
            gc.collect()

            # Get memory usage
            memory_info = self.process.memory_info()
            current_rss = memory_info.rss / 1024 / 1024  # MB

            # Get object counts
            types = objgraph.typestats()
            top_types = sorted(types.items(), key=lambda x: x[1], reverse=True)[:5]

            print(f"\n=== Memory Usage at {location} ===")
            print(
                f"RSS Memory: {current_rss:.1f}MB (Change: {current_rss - self.start_rss:.1f}MB)"
            )
            print("\nTop 5 Object Types:")
            for type_name, count in top_types:
                print(f"{type_name}: {count}")

            # Show what objects are growing
            print("\nGrowing Objects:")
            objgraph.show_growth(limit=3)

            self.last_log_time = current_time


# Create a global tracker
memory_tracker = MemoryTracker()
