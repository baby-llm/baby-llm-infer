import torch
import torch.nn.functional as F
from typing import List, Optional, Set

from .base import TokenSampler
from ...config.generation_config import SamplingConfig

class GreedyTokenSampler(TokenSampler):
    """Samples the token with the highest probability"""
    
    def sample(
        self,
        logits: torch.Tensor,
        sampling_config: SamplingConfig,
        prev_tokens: Optional[List[int]] = None
    ) -> int:
        """Sample the token with the highest probability"""
        # Apply repetition penalty if provided
        if sampling_config.repetition_penalty > 1.0 and prev_tokens:
            self._apply_repetition_penalty(logits, sampling_config.repetition_penalty, prev_tokens)
            
        # Take the most likely token
        return torch.argmax(logits).item()

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        repetition_penalty: float,
        prev_tokens: List[int]
    ) -> None:
        """Apply repetition penalty to logits in-place"""
        if isinstance(prev_tokens, list):
            prev_tokens_set = set(prev_tokens)
        else:
            prev_tokens_set = set(prev_tokens.tolist())
            
        # Apply penalty - reduce probability of tokens that have already appeared
        for token_id in prev_tokens_set:
            logits[token_id] /= repetition_penalty

class TopPTopKSampler(TokenSampler):
    """Samples tokens using temperature, top-p, and top-k sampling"""
    
    def sample(
        self,
        logits: torch.Tensor,
        sampling_config: SamplingConfig,
        prev_tokens: Optional[List[int]] = None
    ) -> int:
        """Sample a token using temperature, top-p, and top-k"""
        # Make a copy of the logits to avoid modifying the original
        working_logits = logits.clone()
        
        # Apply repetition penalty if provided
        if sampling_config.repetition_penalty > 1.0 and prev_tokens:
            self._apply_repetition_penalty(working_logits, sampling_config.repetition_penalty, prev_tokens)
        
        # Apply temperature scaling
        if sampling_config.temperature > 0:
            working_logits = working_logits / sampling_config.temperature
        else:
            # Temperature of 0 means greedy sampling
            return torch.argmax(working_logits).item()
            
        # Apply top-k filtering if specified
        if sampling_config.top_k > 0:
            working_logits = self._apply_top_k(working_logits, sampling_config.top_k)
            
        # Apply top-p filtering    
        if sampling_config.top_p < 1.0:
            working_logits = self._apply_top_p(working_logits, sampling_config.top_p)
            
        # Sample from the filtered distribution
        probs = F.softmax(working_logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        
        return next_token

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        repetition_penalty: float,
        prev_tokens: List[int]
    ) -> None:
        """Apply repetition penalty to logits in-place"""
        if isinstance(prev_tokens, list):
            prev_tokens_set = set(prev_tokens)
        else:
            prev_tokens_set = set(prev_tokens.tolist())
            
        # Apply penalty - reduce probability of tokens that have already appeared
        for token_id in prev_tokens_set:
            logits[token_id] /= repetition_penalty
    
    def _apply_top_k(
        self,
        logits: torch.Tensor,
        top_k: int
    ) -> torch.Tensor:
        """Apply top-k filtering to logits"""
        top_k = min(top_k, logits.size(-1))  # Safety check
        
        # Get top-k values and set others to -inf
        values, _ = torch.topk(logits, top_k)
        min_value = values[-1]
        return torch.where(
            logits < min_value,
            torch.ones_like(logits) * -float('inf'),
            logits
        )
    
    def _apply_top_p(
        self,
        logits: torch.Tensor,
        top_p: float
    ) -> torch.Tensor:
        """Apply top-p (nucleus) filtering to logits"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[0] = False  # Keep at least the top token
        
        # Scatter sorted indices back to original indices
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(0, sorted_indices, sorted_indices_to_remove)
        
        # Set removed indices to -inf and return
        filtered_logits = logits.clone()
        filtered_logits[indices_to_remove] = -float('inf')
        return filtered_logits