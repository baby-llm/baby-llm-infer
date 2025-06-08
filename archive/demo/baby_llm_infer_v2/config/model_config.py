from dataclasses import dataclass, field
from typing import Optional, Union, Dict, Any, List

@dataclass
class QuantizationConfig:
    """Configuration for model quantization"""
    method: str = "none"  # "none", "4bit", or "8bit"
    compute_dtype: str = "float16"
    use_double_quant: bool = True
    quant_type: str = "nf4"
    
    @property
    def is_quantized(self) -> bool:
        return self.method != "none"

@dataclass
class AttentionConfig:
    """Configuration for attention mechanisms"""
    use_optimized: bool = True
    implementation: Optional[str] = None  # "flash_attention_2", "sdpa", or None for default

@dataclass
class ModelConfig:
    """Configuration for model loading and inference"""
    model_name: str
    device: str = "auto"
    trust_remote_code: bool = False
    torch_dtype: str = "auto"  # "auto", "float32", "float16", "bfloat16"
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """Create a ModelConfig from a dictionary"""
        # Extract nested configs
        quant_dict = config_dict.pop("quantization", {})
        attn_dict = config_dict.pop("attention", {})
        
        # Create the main config
        config = cls(**config_dict)
        
        # Set nested configs
        if quant_dict:
            config.quantization = QuantizationConfig(**quant_dict)
        if attn_dict:
            config.attention = AttentionConfig(**attn_dict)
        
        return config