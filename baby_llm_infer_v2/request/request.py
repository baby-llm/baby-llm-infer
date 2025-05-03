import time
import torch
from typing import List, Optional, Dict, Any

from .interfaces import GenerationRequest
from ..config.generation_config import SamplingConfig

class Request(GenerationRequest):
    """Represents a generation request with its state"""
    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_tokens: int = 50,
        sampling_config: Optional[SamplingConfig] = None,
        prompt: str = ""  # Store the original prompt for Qwen models
    ):
        self.input_ids = input_ids
        self.max_tokens = max_tokens
        self.sampling_config = sampling_config or SamplingConfig()
        self.generated_ids: List[int] = []
        self.done = False
        self.start_time = time.time()
        self.attention_mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids)
        self.current_length = input_ids.shape[1]
        self.prompt = prompt  # Store the original prompt
        
    def get_full_sequence(self) -> List[int]:
        """Return the full sequence (input + generated tokens)"""
        return self.input_ids[0].tolist() + self.generated_ids
    
    def add_token(self, token_id: int) -> None:
        """Add a new token to the generated sequence"""
        self.generated_ids.append(token_id)
        
    def is_finished(self, eos_token_id: Optional[int] = None) -> bool:
        """Check if generation should be finished"""
        # Check if we've reached max tokens
        if len(self.generated_ids) >= self.max_tokens:
            return True
            
        # Check if most recent token is EOS
        if eos_token_id is not None and self.generated_ids and self.generated_ids[-1] == eos_token_id:
            return True
            
        return False