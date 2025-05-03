from abc import ABC, abstractmethod
import torch
from typing import List, Optional

from ...config.generation_config import SamplingConfig

class TokenSampler(ABC):
    """Base interface for token sampling strategies"""
    
    @abstractmethod
    def sample(
        self,
        logits: torch.Tensor,
        sampling_config: SamplingConfig,
        prev_tokens: Optional[List[int]] = None
    ) -> int:
        """Sample a token from logits based on the sampling configuration"""
        pass