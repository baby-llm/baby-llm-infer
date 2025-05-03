from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Any, Dict

class Cache(ABC):
    """Abstract interface for caching mechanisms"""
    
    @abstractmethod
    def get(self) -> Any:
        """Get the stored cache"""
        pass
    
    @abstractmethod
    def update(self, new_cache: Any) -> None:
        """Update the stored cache"""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the cache to empty state"""
        pass

class GenerationRequest(ABC):
    """Abstract interface for generation requests"""
    
    @abstractmethod
    def get_full_sequence(self) -> List[int]:
        """Get the full token sequence (input + generated tokens)"""
        pass
    
    @abstractmethod
    def add_token(self, token_id: int) -> None:
        """Add a new token to the generated sequence"""
        pass
    
    @abstractmethod
    def is_finished(self, eos_token_id: Optional[int] = None) -> bool:
        """Check if generation should be finished"""
        pass