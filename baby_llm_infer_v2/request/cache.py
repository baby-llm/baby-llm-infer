from typing import Optional, Tuple, Any

from .interfaces import Cache

class KeyValueCache(Cache):
    """Optimized key-value cache for transformer layers"""
    def __init__(self):
        self.past_key_values = None
    
    def get(self) -> Optional[Tuple]:
        """Get the stored KV cache"""
        return self.past_key_values
    
    def update(self, past_key_values: Tuple) -> None:
        """Update the stored KV cache"""
        self.past_key_values = past_key_values
        
    def reset(self) -> None:
        """Reset the KV cache"""
        self.past_key_values = None