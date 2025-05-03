import torch
import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

from ..config.generation_config import GenerationConfig, SamplingConfig
from ..request.request import Request
from ..request.batcher import ContinuousBatcher
from .sampling.strategies import TopPTopKSampler, GreedyTokenSampler

logger = logging.getLogger('optimized_inference')

class GenerationEngine:
    """Engine for text generation with various optimization strategies"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.samplers = {
            'greedy': GreedyTokenSampler(),
            'sampling': TopPTopKSampler()
        }
    
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> Dict[str, Any]:
        """Generate text from a prompt
        
        Args:
            prompt: The input text prompt
            config: Generation configuration
            
        Returns:
            Dictionary with generated text and metrics
        """
        # Use default config if none provided
        if config is None:
            config = GenerationConfig()
        
        # Get device for the model
        device = self._get_model_device()
        
        # Encode the prompt
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device) if hasattr(encoded, 'attention_mask') else torch.ones_like(input_ids)
        
        # Time the generation
        start_time = time.time()
        
        if config.use_kv_cache:
            # Use optimized generation with KV cache
            generated_ids, _ = self._generate_with_kv_cache(
                input_ids,
                attention_mask,
                config
            )
        else:
            # Use standard generation without KV cache
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=config.max_tokens,
                    do_sample=config.sampling.temperature > 0,
                    temperature=config.sampling.temperature,
                    top_p=config.sampling.top_p,
                    top_k=config.sampling.top_k if config.sampling.top_k > 0 else None,
                    repetition_penalty=config.sampling.repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
            # Extract generated tokens (excluding input tokens)
            generated_ids = outputs[0, input_ids.shape[1]:].tolist()
        
        # Combine input and generated tokens
        all_ids = input_ids[0].tolist() + generated_ids
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(all_ids, skip_special_tokens=True)
        
        # Calculate generation statistics
        total_time = time.time() - start_time
        tokens_per_second = len(generated_ids) / max(0.001, total_time)
        
        return {
            "text": generated_text,
            "metrics": {
                "input_tokens": len(input_ids[0]),
                "generated_tokens": len(generated_ids),
                "total_tokens": len(all_ids),
                "generation_time": total_time,
                "tokens_per_second": tokens_per_second
            }
        }
    
    def generate_batch(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None
    ) -> List[Dict[str, Any]]:
        """Generate text for multiple prompts using continuous batching
        
        Args:
            prompts: List of input text prompts
            config: Generation configuration
            
        Returns:
            List of dictionaries with generated text and metrics
        """
        # Use default config if none provided
        if config is None:
            config = GenerationConfig()
            
        # Get device for the model
        device = self._get_model_device()
        
        # Create the continuous batcher
        batcher = ContinuousBatcher(self.model, self.tokenizer, max_batch_size=len(prompts))
        
        # Add all requests to the batcher
        start_time = time.time()
        
        for i, prompt in enumerate(prompts):
            encoded = self.tokenizer(prompt, return_tensors="pt")
            input_ids = encoded.input_ids.to(device)
            
            request = Request(
                input_ids, 
                max_tokens=config.max_tokens,
                sampling_config=config.sampling
            )
            batcher.add_request(i, request)
        
        # Wait for all requests to complete
        results = {}
        all_done = False
        
        while not all_done:
            all_done = True
            for i in range(len(prompts)):
                if i not in results:
                    result = batcher.get_result(i)
                    if result is not None:
                        # Get the input tokens
                        input_tokens = self.tokenizer(prompts[i], return_tensors="pt").input_ids[0].tolist()
                        
                        # Combine input and generated tokens
                        all_tokens = input_tokens + result
                        
                        # Decode the final text
                        results[i] = {
                            "text": self.tokenizer.decode(all_tokens, skip_special_tokens=True),
                            "metrics": {
                                "input_tokens": len(input_tokens),
                                "generated_tokens": len(result),
                                "total_tokens": len(all_tokens)
                            }
                        }
                    else:
                        all_done = False
            
            if not all_done:
                time.sleep(0.1)
        
        # Stop the batcher thread
        batcher.stop()
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Update metrics with timing information
        for result in results.values():
            result["metrics"]["generation_time"] = total_time
            result["metrics"]["tokens_per_second"] = result["metrics"]["generated_tokens"] / total_time
        
        # Return results in the same order as prompts
        return [results[i] for i in range(len(prompts))]
    
    def _get_model_device(self) -> torch.device:
        """Determine device where model is located"""
        if hasattr(self.model, 'device'):
            return self.model.device
        
        # If model is spread across devices, use the first parameter's device
        return next(self.model.parameters()).device
    
    def _generate_with_kv_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        config: GenerationConfig
    ) -> Tuple[List[int], float]:
        """Generate text using KV cache for efficient token generation"""
        # Select the appropriate sampler
        if config.sampling.temperature == 0:
            sampler = self.samplers['greedy']
        else:
            sampler = self.samplers['sampling']
        
        # Record start time
        start_time = time.time()
        generated_ids = []
        
        with torch.no_grad():
            # Process the full prompt first
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True
            )
            
            # Get next token logits
            next_token_logits = outputs.logits[0, -1, :]
            
            # Apply repetition penalty to input tokens
            full_sequence = input_ids[0].tolist()
            
            # Sample next token
            next_token = sampler.sample(
                next_token_logits,
                config.sampling,
                full_sequence
            )
            generated_ids.append(next_token)
            
            # Get KV cache
            past_key_values = outputs.past_key_values
            
            # Check for EOS token
            if next_token == self.tokenizer.eos_token_id:
                pass  # We'll still include this EOS token but won't generate further
            
            # Generate remaining tokens auto-regressively
            for _ in range(config.max_tokens - 1):
                # Check if we've already hit EOS
                if generated_ids[-1] == self.tokenizer.eos_token_id:
                    break
                    
                # Prepare inputs for the next iteration
                new_input_ids = torch.tensor([[next_token]], device=input_ids.device)
                
                # Extend attention mask for the new token
                attention_mask = torch.cat([
                    attention_mask, 
                    torch.ones((1, 1), device=input_ids.device, dtype=attention_mask.dtype)
                ], dim=1)
                
                # Forward pass with KV cache
                outputs = self.model(
                    input_ids=new_input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                
                # Get full sequence for repetition penalty
                full_sequence = input_ids[0].tolist() + generated_ids
                
                # Get logits and sample next token
                next_token_logits = outputs.logits[0, -1, :]
                next_token = sampler.sample(
                    next_token_logits,
                    config.sampling,
                    full_sequence
                )
                generated_ids.append(next_token)
                
                # Check for EOS token
                if next_token == self.tokenizer.eos_token_id:
                    break
                
                # Update KV cache
                past_key_values = outputs.past_key_values
        
        total_time = time.time() - start_time
        return generated_ids, total_time