#!/usr/bin/env python3
"""Opt-in timing wrapper; executes the unchanged main entry and scheduling policy.

Top-level installation also runs in the spawned compute worker. Use only in a
separate diagnostic run: never mix instrumented timings into the main matrix.
"""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.schedulers import stream_scheduler


class TimedStreamScheduler(stream_scheduler.StreamSchedulerPlugin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_started = None
        self.epoch_stalls = []

    def before_training_epoch(self, strategy, **kwargs):
        self.epoch_started = time.perf_counter()

    def before_training_iteration(self, strategy, **kwargs):
        if self.epoch_started is not None:
            now = time.perf_counter()
            self.epoch_stalls.append({
                'train_updates': self.steps,
                'before_epoch_to_first_iteration_ms': (now-self.epoch_started)*1000,
                'previous_update_to_first_iteration_ms':
                    (now-self.last_train_end)*1000 if self.last_train_end is not None else None})
            self.epoch_started = None
        super().before_training_iteration(strategy, **kwargs)

    def finish(self, strategy):
        result = super().finish(strategy)
        Path('stream_epoch_stalls.json').write_text(json.dumps(self.epoch_stalls, indent=2))
        print('[EpochStartup] '+json.dumps(self.epoch_stalls), flush=True)
        return result


stream_scheduler.StreamSchedulerPlugin = TimedStreamScheduler

if __name__ == '__main__':
    import main
    main.main()
