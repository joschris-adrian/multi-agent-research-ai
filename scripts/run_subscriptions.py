import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.scheduler.scheduler import run_due_subscriptions

if __name__ == "__main__":
    run_due_subscriptions()