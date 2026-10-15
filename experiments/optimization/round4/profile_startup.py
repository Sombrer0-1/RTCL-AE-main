"""Profile only epoch startup; these timings are diagnostic, not benchmark data."""
import cProfile
import gc
import io
import json
from pathlib import Path
import pstats
import sys
import time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from src.schedulers import stream_scheduler

class ProfileStream(stream_scheduler.StreamSchedulerPlugin):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.profiler=cProfile.Profile()
        self.gc_events=[]
        self.gc_started={}
        self.in_startup=False
        gc.callbacks.append(self.gc_event)

    def gc_event(self,phase,info):
        gen=info['generation']
        if phase=='start':
            self.gc_started[gen]=time.perf_counter()
        else:
            self.gc_events.append(dict(generation=gen,seconds=time.perf_counter()-self.gc_started[gen],
                                       startup=self.in_startup,steps=self.steps,collected=info['collected']))

    def before_training_epoch(self,strategy,**kwargs):
        self.in_startup=True
        self.profiler.enable()

    def before_training_iteration(self,strategy,**kwargs):
        if self.in_startup:
            self.profiler.disable()
            self.in_startup=False
        super().before_training_iteration(strategy,**kwargs)

    def finish(self,strategy):
        result=super().finish(strategy)
        self.profiler.disable()
        s=io.StringIO()
        pstats.Stats(self.profiler,stream=s).sort_stats('cumulative').print_stats(45)
        Path('stream_startup_profile.txt').write_text(s.getvalue())
        Path('stream_gc_events.json').write_text(json.dumps(self.gc_events,indent=2))
        gc.callbacks.remove(self.gc_event)
        return result

stream_scheduler.StreamSchedulerPlugin=ProfileStream
if __name__=='__main__':
    import main
    main.main()
