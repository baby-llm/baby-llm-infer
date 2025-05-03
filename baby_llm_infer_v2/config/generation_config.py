from dataclasses import dataclass
from typing import Optional, Dict, Any, List

@dataclass
class SamplingConfig:
    """Configuration for token sampling"""
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 0
    repetition_penalty: float = 1.0
    
    @classmethod
    def greedy(cls) -> "SamplingConfig":
        """Create a configuration for greedy sampling"""
        return cls(temperature=0.0, top_p=1.0, top_k=1, repetition_penalty=1.0)

@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_tokens: int = 50
    sampling: SamplingConfig = SamplingConfig()
    use_kv_cache: bool = True
    stop_sequences: List[str] = None
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GenerationConfig":
        """Create a GenerationConfig from a dictionary"""
        # Extract nested configs
        sampling_dict = config_dict.pop("sampling", {})
        
        # Create the main config
        config = cls(**config_dict)
        
        # Set nested configs
        if sampling_dict:
            config.sampling = SamplingConfig(**sampling_dict)
        
        return config