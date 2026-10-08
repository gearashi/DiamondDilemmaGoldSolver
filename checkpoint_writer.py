"""One coherent GPU snapshot at a time, with disk writing off the search thread."""
from concurrent.futures import ThreadPoolExecutor
import time

class CheckpointWriter:
    def __init__(self, snapshot, write, *, clock=time.monotonic):
        self.snapshot=snapshot
        self.write=write
        self.clock=clock
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='diamond-save')
        self.future=None
        self.pending_generation=None
        self.completed_generation=None
        self.last_completed_at=clock()
        self.info={'phase':'idle','completed_generation':None}
        self.closed=False

    def poll(self, *, wait=False):
        if self.future is None or (not wait and not self.future.done()):
            return False
        result=self.future.result()
        self.completed_generation=self.pending_generation
        self.last_completed_at=self.clock()
        self.info.update(result or {})
        self.info.update(phase='idle',completed_generation=self.completed_generation)
        self.future=None
        self.pending_generation=None
        return True

    def request(self, generation, *, minimum_interval=0):
        if self.closed:raise RuntimeError('Checkpoint writer is closed')
        self.poll()
        if (self.future is not None or self.completed_generation==generation
                or self.clock()-self.last_completed_at<minimum_interval):
            return False
        started=self.clock()
        self.info.update(phase='capturing',generation=generation)
        snapshot=self.snapshot()
        self.info.update(phase='saving',capture_seconds=self.clock()-started)
        self.pending_generation=generation
        self.future=self.executor.submit(self.write,snapshot,generation)
        return True

    def finish(self, generation):
        """Wait for any older save, then save the final generation exactly once."""
        self.poll(wait=True)
        self.request(generation)
        self.poll(wait=True)
        return dict(self.info)

    def close(self):
        if not self.closed:
            self.closed=True
            self.executor.shutdown(wait=True)
