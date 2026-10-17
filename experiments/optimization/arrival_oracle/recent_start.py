#!/usr/bin/env python3
"""Isolate the window planner's cost history without changing the active matrix."""
from collections import deque
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import main as entry
import src.schedulers.stream_scheduler as stream


class RecentPlanner(stream.StreamSchedulerPlugin):
    def _planned_wait(self, elapsed):
        # Only the planning point estimate uses 16 observations. The common
        # safety envelope and quota controller retain their original 64 entries.
        original = self.compute_costs
        self.compute_costs = deque(list(original)[-16:], maxlen=16)
        try:
            return super()._planned_wait(elapsed)
        finally:
            self.compute_costs = original


stream.StreamSchedulerPlugin = RecentPlanner

if __name__ == '__main__':
    entry.main()
