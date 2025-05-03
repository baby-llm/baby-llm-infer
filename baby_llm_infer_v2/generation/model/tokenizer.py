from transformers import AutoTokenizer
from typing import Optional, List, Union
import logging

logger = logging.getLogger('optimized_inference')

class TokenizerLoader:
    """Utility class for loading and configuring tokenizers"""
    
    @staticmethod
    def load_tokenizer(
        model_name: str,
        trust_remote_code: bool = False,
        default_eos_token: str = "</s>"
    ):
        """Load tokenizer with appropriate settings
        
        Args:
            model_name: HuggingFace model name or path
            trust_remote_code: Whether to trust remote code
            default_eos_token: Default EOS token if none is found
            
        Returns:
            Configured tokenizer
        """
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=trust_remote_code
        )
        
        # Check if this is a Qwen model, which requires left padding with Flash Attention
        is_qwen_model = "qwen" in model_name.lower()
        
        # Apply appropriate padding side based on model type
        if is_qwen_model:
            tokenizer.padding_side = "left"
            logger.info(f"Qwen model detected: Setting tokenizer padding_side to left for compatibility")
        
        logger.info(f"Tokenizer padding side: {tokenizer.padding_side}")
        
        # Ensure padding token exists
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token = tokenizer.eos_token = default_eos_token
        
        return tokenizer