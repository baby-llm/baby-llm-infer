from transformers import AutoTokenizer
from typing import Optional, List, Union

class TokenizerLoader:
    """Utility class for loading and configuring tokenizers"""
    
    @staticmethod
    def load_tokenizer(
        model_name: str,
        trust_remote_code: bool = False,
        padding_side: str = "left",
        default_eos_token: str = "</s>"
    ):
        """Load tokenizer with appropriate settings
        
        Args:
            model_name: HuggingFace model name or path
            trust_remote_code: Whether to trust remote code
            padding_side: Side to add padding ("left" or "right")
            default_eos_token: Default EOS token if none is found
            
        Returns:
            Configured tokenizer
        """
        # Load tokenizer with appropriate settings
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=trust_remote_code,
            padding_side=padding_side
        )
        
        # Ensure padding token exists
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token = tokenizer.eos_token = default_eos_token
        
        return tokenizer