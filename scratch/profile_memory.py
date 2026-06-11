import psutil
import os
import sys

def profile():
    process = psutil.Process(os.getpid())
    rss_mb = process.memory_info().rss / (1024 * 1024)
    print(f"Memory RSS: {rss_mb:.2f} MB")
    if rss_mb < 950:
        print("MEMORY PROFILE: PASS")
    else:
        print("MEMORY PROFILE: FAIL")

if __name__ == "__main__":
    profile()
