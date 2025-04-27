# scheduler.py
from collections import deque
from typing import List, Deque, Tuple
from sequence import Sequence, SequenceStatus
from kv_cache import KVCacheManager
from utils import logger
import time

class SchedulerConfig:
    def __init__(self, max_batch_size: int = 8):
        self.max_batch_size = max_batch_size
        # Could add more policies here (e.g., max sequence length)

class Scheduler:
    """
    Implements simplified continuous batching.
    Manages waiting and running queues, schedules sequences for the next step.
    """
    def __init__(self, config: SchedulerConfig, cache_manager: KVCacheManager):
        self.config = config
        self.cache_manager = cache_manager
        self.waiting_queue: Deque[Sequence] = deque()
        self.running_pool: Dict[str, Sequence] = {} # request_id -> Sequence

    def add_sequence(self, sequence: Sequence):
        """Adds a new sequence request to the waiting queue."""
        logger.info(f"Adding sequence {sequence.request_id} to waiting queue.")
        self.waiting_queue.append(sequence)

    def schedule(self) -> List[Sequence]:
        """
        The core scheduling logic for one step.
        1. Remove finished sequences from the running pool.
        2. Promote waiting sequences to running if space allows.
        3. Return the list of sequences to run in the current step.
        """
        now = time.time()

        # 1. Process running pool: remove finished, free resources
        finished_ids = []
        for seq_id, seq in self.running_pool.items():
            if seq.is_finished():
                logger.info(f"Sequence {seq_id} finished ({seq.status.name}). Removing from running pool.")
                finished_ids.append(seq_id)
                self.cache_manager.free_for_sequence(seq) # Free KV cache

        for seq_id in finished_ids:
            del self.running_pool[seq_id]

        # 2. Promote waiting sequences to running pool if space allows
        while self.waiting_queue and len(self.running_pool) < self.config.max_batch_size:
            seq = self.waiting_queue.popleft()
            seq.status = SequenceStatus.RUNNING
            seq.last_scheduled_time = now
            self.cache_manager.allocate_for_sequence(seq) # Allocate KV cache entry
            self.running_pool[seq.request_id] = seq
            logger.info(f"Promoted sequence {seq.request_id} to running pool.")

        # 3. Return the current batch of running sequences
        active_batch = list(self.running_pool.values())
        # Update scheduled time for active sequences
        for seq in active_batch:
             seq.last_scheduled_time = now

        if active_batch:
            logger.debug(f"Scheduler returning batch of size {len(active_batch)}")
        # else:
            # logger.debug("Scheduler returning empty batch.")

        return active_batch

    def has_unfinished_requests(self) -> bool:
        """Checks if there are any requests waiting or running."""
        return bool(self.waiting_queue or self.running_pool)