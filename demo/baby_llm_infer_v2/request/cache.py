from typing import Optional, Tuple, Any

from .interfaces import Cache

class KeyValueCache(Cache):
    """Optimized key-value cache for transformer layers"""
    def __init__(self, is_qwen_model=False):
        self.past_key_values = None
        self.is_qwen_model = is_qwen_model
    
    def get(self) -> Optional[Any]:
        """Get the stored KV cache"""
        return self.past_key_values
    
    def update(self, past_key_values: Any) -> None:
        """Update the stored KV cache
        
        For Qwen3 models, past_key_values should be passed through directly
        For other models, past_key_values is typically a tuple structure
        """
        # Store as-is - we'll handle the format differences in the batcher
        self.past_key_values = past_key_values
        
    def reset(self) -> None:
        """Reset the KV cache"""
        self.past_key_values = None