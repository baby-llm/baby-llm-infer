# kv_cache.py
import torch
from typing import Dict, List, Tuple, Optional
from utils import logger

# Type hint for HuggingFace past_key_values structure:
# Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
PastKeyValueType = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]

class KVCacheManager:
    """
    Manages KV caches for sequences, simulating dynamic allocation/deallocation
    inspired by PagedAttention (without actual paging/blocks in this MVP).
    It stores the *entire* past_key_values structure per sequence.
    """
    def __init__(self, config, device):
        self.cache: Dict[int, Optional[PastKeyValueType]] = {} # seq.kv_cache_handle -> KV Cache
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.device = device
        self._next_handle = 0
        logger.info(f"KV Cache Manager initialized for {self.num_layers} layers.")

    def _allocate_handle(self) -> int:
        """Gets a unique ID for a new sequence's cache."""
        handle = self._next_handle
        self._next_handle += 1
        self.cache[handle] = None # Initialize as empty
        # logger.debug(f"Allocated KV cache handle: {handle}")
        return handle

    def get_cache(self, handle: int) -> Optional[PastKeyValueType]:
        """Retrieves the KV cache for a given handle."""
        return self.cache.get(handle)

    def update_cache(self, handle: int, new_past_key_values: PastKeyValueType):
        """
        Updates the cache for a sequence.
        In this simplified version, we just replace the old cache with the new one
        returned by the model, which includes the appended keys/values.
        A more block-like simulation would involve appending to tensors.
        """
        if handle not in self.cache:
            logger.warning(f"Attempted to update non-existent KV cache handle: {handle}")
            return

        # Basic validation (optional)
        # assert len(new_past_key_values) == self.num_layers
        # assert new_past_key_values[0][0].shape[0] == 1 # Assuming update is per-sequence

        self.cache[handle] = new_past_key_values
        # logger.debug(f"Updated KV cache for handle: {handle}")


    def allocate_for_sequence(self, sequence):
        """Allocates a handle and cache entry for a new sequence."""
        if sequence.kv_cache_handle is not None:
            logger.warning(f"Sequence {sequence.request_id} already has a KV cache handle.")
            return
        handle = self._allocate_handle()
        sequence.kv_cache_handle = handle
        logger.info(f"Allocated KV cache handle {handle} for Seq {sequence.request_id}")

    def free(self, handle: int):
        """Frees the cache associated with a handle (when sequence finishes)."""
        if handle in self.cache:
            del self.cache[handle]
            # logger.info(f"Freed KV cache for handle: {handle}")
        # else:
            # logger.warning(f"Attempted to free non-existent KV cache handle: {handle}")

    def free_for_sequence(self, sequence):
        """Frees the cache associated with a sequence."""
        if sequence.kv_cache_handle is not None:
            self.free(sequence.kv_cache_handle)
            sequence.kv_cache_handle = None # Important to clear handle on sequence object