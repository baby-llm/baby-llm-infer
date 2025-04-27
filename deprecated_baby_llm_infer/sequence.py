import time
from enum import Enum
from typing import List, Optional, Tuple

class SequenceStatus(Enum):
    WAITING = 1
    RUNNING = 2
    FINISHED_STOPPED = 3 # Finished by EOS or stop token
    FINISHED_LENGTH = 4  # Finished by reaching max length

class Sequence:
    """Represents a single request/sequence being processed."""
    def __init__(self, request_id: str, prompt: str, prompt_token_ids: List[int], max_new_tokens: int):
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.output_token_ids: List[int] = []
        self.status = SequenceStatus.WAITING
        self.max_new_tokens = max_new_tokens
        self.created_time = time.time()
        self.last_scheduled_time = None

        # Placeholder for KV cache handle - managed externally
        self.kv_cache_handle = None # Could be an ID or direct reference

    def get_full_sequence_ids(self) -> List[int]:
        """Returns the prompt + generated tokens."""
        return self.prompt_token_ids + self.output_token_ids

    def get_len(self) -> int:
        """Current total length of the sequence."""
        return len(self.get_full_sequence_ids())

    def is_finished(self) -> bool:
        return self.status in [SequenceStatus.FINISHED_STOPPED, SequenceStatus.FINISHED_LENGTH]

    def append_token_id(self, token_id: int):
        """Appends a generated token."""
        self.output_token_ids.append(token_id)

    def __repr__(self):
        return (f"Sequence(id={self.request_id}, status={self.status.name}, "
                f"len={self.get_len()}, output_len={len(self.output_token_ids)})")

# Optional: Grouping sequences if doing beam search etc. For MVP, one seq per request.
# class SequenceGroup:
#     def __init__(self, request_id: str):
#         self.request_id = request_id
#         self.seqs: List[Sequence] = []
#         # ... other group metadata