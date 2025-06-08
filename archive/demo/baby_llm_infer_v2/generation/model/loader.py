import torch
import logging
from typing import Optional, Dict, Any, Union
import flash_attn

from transformers import AutoModelForCausalLM

from ...config.model_config import ModelConfig, QuantizationConfig, AttentionConfig
from ...utils.device import get_optimal_device, get_torch_dtype

logger = logging.getLogger('optimized_inference')

class ModelFactory:
    """Factory class for creating optimized models"""
    
    @classmethod
    def create_model(cls, config: ModelConfig) -> AutoModelForCausalLM:
        """Create a model from configuration"""
        logger.info(f"Loading model {config.model_name}...")
        
        # Ensure we have a valid device
        device = get_optimal_device(config.device)
        logger.info(f"Using device: {device}")
        
        # Prepare model loading arguments
        model_kwargs = cls._prepare_model_kwargs(config, device)
        
        # Load the model
        try:
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                **model_kwargs
            )
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        
        # If not using device_map="auto", move model to the specified device
        if model_kwargs.get("device_map") is None:
            model.to(device)
        
        model.eval()
        logger.info(f"Loaded model with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
        
        # Suggest torch.compile for advanced users
        if hasattr(torch, 'compile') and torch.__version__ >= "2.0.0":
            logger.info("Note: For PyTorch >= 2.0, you can further optimize with: model = torch.compile(model, mode='reduce-overhead')")
        
        return model
    
    @classmethod
    def _prepare_model_kwargs(cls, config: ModelConfig, device: str) -> Dict[str, Any]:
        """Prepare kwargs for model loading with optimizations"""
        # Get torch dtype
        dtype = get_torch_dtype(config.torch_dtype, device)
        
        # Prepare basic kwargs
        model_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": config.trust_remote_code,
        }
        
        # Add quantization if requested
        quantization_config = cls._prepare_quantization_config(config.quantization)
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        # Add attention implementation if available and requested
        attention_impl = cls._prepare_attention_implementation(config.attention)
        if attention_impl:
            model_kwargs["attn_implementation"] = attention_impl
        
        # For large models, use device_map="auto" for automatic offloading
        if not config.quantization.is_quantized and cls._is_large_model(config.model_name):
            model_kwargs["device_map"] = "auto"
            logger.info("Using automatic device mapping for large model")
        else:
            # For smaller models or when using quantization, we can specify the device directly
            model_kwargs["device_map"] = None
        
        return model_kwargs
    
    @staticmethod
    def _is_large_model(model_name: str) -> bool:
        """Check if model is likely a large model that needs device mapping"""
        large_model_keywords = ["llama", "qwen", "mistral", "falcon", "mpt", "bloom"]
        return any(keyword in model_name.lower() for keyword in large_model_keywords)
    
    @classmethod
    def _prepare_quantization_config(cls, config: QuantizationConfig) -> Optional[Any]:
        """Prepare quantization configuration"""
        if not config.is_quantized:
            return None
            
        # Check for BitsAndBytes support
        try:
            from transformers import BitsAndBytesConfig
            has_bnb_config = True
        except ImportError:
            has_bnb_config = False
            logger.warning("BitsAndBytesConfig not available, quantization disabled")
            return None
        
        try:
            import bitsandbytes as bnb
            has_bitsandbytes = True
        except ImportError:
            has_bitsandbytes = False
            logger.warning("BitsAndBytes not available, quantization disabled")
            return None
        
        # Create appropriate configuration
        if config.method == "4bit" and has_bnb_config:
            logger.info("Using 4-bit quantization")
            compute_dtype = torch.float16  # Default
            if config.compute_dtype == "bfloat16" and hasattr(torch, "bfloat16"):
                compute_dtype = torch.bfloat16
            
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=config.use_double_quant,
                bnb_4bit_quant_type=config.quant_type
            )
        elif config.method == "8bit" and has_bnb_config and has_bitsandbytes:
            logger.info("Using 8-bit quantization")
            return BitsAndBytesConfig(load_in_8bit=True)
        
        logger.warning(f"Unsupported quantization method: {config.method}")
        return None
    
    @classmethod
    def _prepare_attention_implementation(cls, config: AttentionConfig) -> Optional[str]:
        """Determine the optimal attention implementation"""
        if not config.use_optimized:
            return None
            
        # If implementation is explicitly set, try to use it
        if config.implementation:
            if config.implementation == "flash_attention_2":
                try:
                    import flash_attn
                    logger.info("Using Flash Attention 2 as requested")
                    return "flash_attention_2"
                except ImportError:
                    logger.warning("Flash Attention 2 requested but not available")
            elif config.implementation == "sdpa":
                if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                    logger.info("Using PyTorch SDPA as requested")
                    return "sdpa"
                else:
                    logger.warning("PyTorch SDPA requested but not available")
            return None
        
        # Auto-detect best available attention implementation
        try:
            import flash_attn
            logger.info("Using Flash Attention 2 for improved performance")
            return "flash_attention_2"
        except ImportError:
            pass
            
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            logger.info("Using PyTorch's Scaled Dot Product Attention")
            return "sdpa"
            
        logger.info("Using default attention implementation")
        return None