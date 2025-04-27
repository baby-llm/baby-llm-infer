# huggingface_optimized_inference.py
import torch
import torch.nn.functional as F
import time
import argparse
from queue import Queue
from threading import Thread
from threading import Lock
import os
import logging

from transformers import AutoTokenizer, AutoModelForCausalLM

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import optimizations
try:
    import bitsandbytes as bnb
    has_bitsandbytes = True
    logger.info("BitsAndBytes quantization available")
except ImportError:
    has_bitsandbytes = False
    logger.info("BitsAndBytes not available, quantization options will be limited")

try:
    from transformers import BitsAndBytesConfig
    has_bnb_config = True
except ImportError:
    has_bnb_config = False

try:
    import flash_attn
    has_flash_attn = True
    logger.info("Flash Attention 2 available")
except ImportError:
    has_flash_attn = False
    logger.info("Flash Attention not available")

###################
# MODEL LOADING UTILITIES
###################

def load_model_and_tokenizer(model_name, device, quantize=None, trust_remote_code=False, use_optimized_attention=True):
    """Load model with appropriate optimizations based on size and capabilities"""
    logger.info(f"Loading model {model_name}...")
    
    # Determine dtype
    if device == "cpu" or quantize:
        dtype = torch.float32  # Will be overridden by quantization if enabled
    else:
        dtype = torch.float16
    
    # Configure quantization
    quantization_config = None
    if quantize and has_bnb_config:
        if quantize == "4bit":
            logger.info("Using 4-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif quantize == "8bit" and has_bitsandbytes:
            logger.info("Using 8-bit quantization")
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    # Try to enable optimized attention if requested
    attn_implementation = None
    if use_optimized_attention:
        if has_flash_attn:
            attn_implementation = "flash_attention_2"
            logger.info("Using Flash Attention 2 for improved performance")
        elif hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            attn_implementation = "sdpa"
            logger.info("Using PyTorch's Scaled Dot Product Attention")
        else:
            logger.info("Using default attention implementation")
    
    # Load tokenizer first with appropriate settings
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=trust_remote_code,
        padding_side="left"  # Important for efficient batch processing
    )
    
    # Ensure padding token exists
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.eos_token = "</s>"
    
    # Load model with optimizations
    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
    }
    
    # Add attention implementation if available and requested
    if attn_implementation and use_optimized_attention:
        model_kwargs["attn_implementation"] = attn_implementation
    
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
    
    # For large models, use device_map="auto" for automatic offloading
    if not quantize and "llama" in model_name.lower() or "qwen" in model_name.lower() or "mistral" in model_name.lower():
        model_kwargs["device_map"] = "auto"
        logger.info("Using automatic device mapping for large model")
    else:
        # For smaller models or when using quantization, we can specify the device directly
        model_kwargs["device_map"] = None
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    # If not using device_map="auto", move model to the specified device
    if model_kwargs["device_map"] is None:
        model.to(device)
    
    model.eval()
    logger.info(f"Loaded model with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    
    # Suggest torch.compile for advanced users
    if hasattr(torch, 'compile') and torch.__version__ >= "2.0.0":
        logger.info("Note: For PyTorch >= 2.0, you can further optimize with: model = torch.compile(model, mode='reduce-overhead')")
    
    return model, tokenizer

###################
# KV CACHE
###################

class KeyValueCache:
    """Optimized key-value cache for transformer layers"""
    def __init__(self):
        self.past_key_values = None
    
    def get(self):
        return self.past_key_values
    
    def update(self, past_key_values):
        self.past_key_values = past_key_values
        
    def reset(self):
        self.past_key_values = None

###################
# REQUEST HANDLING
###################

class Request:
    """Represents a generation request with its state"""
    def __init__(self, input_ids, max_tokens=50, temperature=1.0, top_p=0.9, top_k=0, repetition_penalty=1.0):
        self.input_ids = input_ids
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty  # New parameter
        self.generated_ids = []
        self.done = False
        self.start_time = time.time()
        self.attention_mask = torch.ones_like(input_ids)
        self.current_length = input_ids.shape[1]  # Track sequence length
        
    def get_full_sequence(self):
        """Return the full sequence (input + generated tokens)"""
        return self.input_ids[0].tolist() + self.generated_ids
        
    def is_finished(self, eos_token_id=None):
        """Check if generation should be finished"""
        # Check if we've reached max tokens
        if len(self.generated_ids) >= self.max_tokens:
            return True
            
        # Check if most recent token is EOS
        if eos_token_id is not None and self.generated_ids and self.generated_ids[-1] == eos_token_id:
            return True
            
        return False

class ContinuousBatcher:
    """Manages batched inference across multiple requests (core optimization)"""
    def __init__(self, model, tokenizer, max_batch_size=8):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.request_queue = Queue()
        self.active_requests = {}
        self.lock = Lock()
        self.running = True
        
        # Maintain a KV cache for each active request
        self.request_caches = {}
        
        # Start processing thread
        self.thread = Thread(target=self._process_batches)
        self.thread.daemon = True
        self.thread.start()
        
    def add_request(self, request_id, request):
        """Add a new request to be processed"""
        with self.lock:
            # Create a new KV cache for this request
            self.request_caches[request_id] = KeyValueCache()
        self.request_queue.put((request_id, request))
        
    def get_result(self, request_id):
        """Get results for a specific request if completed"""
        with self.lock:
            if request_id in self.active_requests and self.active_requests[request_id].done:
                request = self.active_requests[request_id]
                # Clean up caches
                if request_id in self.request_caches:
                    del self.request_caches[request_id]
                del self.active_requests[request_id]
                return request.generated_ids
        return None
        
    def _process_batches(self):
        """Main processing loop - continuously builds and processes batches"""
        while self.running:
            # Collect requests to batch together
            requests_to_process = []
            
            # Get new requests from queue
            while not self.request_queue.empty() and len(requests_to_process) < self.max_batch_size:
                request_id, request = self.request_queue.get()
                with self.lock:
                    self.active_requests[request_id] = request
                requests_to_process.append((request_id, request))
                
            # Add existing unfinished requests
            with self.lock:
                for request_id, request in list(self.active_requests.items()):
                    if not request.done and (request_id, request) not in requests_to_process:
                        requests_to_process.append((request_id, request))
                        if len(requests_to_process) >= self.max_batch_size:
                            break
            
            if not requests_to_process:
                time.sleep(0.01)  # Prevent CPU spinning
                continue
                
            # Process the batch
            self._process_batch(requests_to_process)
            
    def _prepare_model_inputs(self, requests_to_process):
        """Prepare unified inputs for model forward pass"""
        # Group requests by cache status (fresh vs continuation)
        fresh_requests = []
        continuation_requests = []
        
        for request_id, request in requests_to_process:
            # Check if this request has a KV cache already
            has_cache = False
            with self.lock:
                if request_id in self.request_caches:
                    kv_cache = self.request_caches[request_id].get()
                    has_cache = kv_cache is not None
            
            if has_cache:
                continuation_requests.append((request_id, request))
            else:
                fresh_requests.append((request_id, request))
        
        return {
            "fresh_requests": fresh_requests,
            "continuation_requests": continuation_requests
        }
            
    def _process_batch(self, requests):
        """Process a batch of requests together (key to efficient inference)"""
        # Prepare and categorize batch inputs
        request_groups = self._prepare_model_inputs(requests)
        
        # Process fresh requests (no KV cache) in a batch
        if request_groups["fresh_requests"]:
            batch_input_ids = []
            batch_attention_masks = []
            
            for request_id, request in request_groups["fresh_requests"]:
                batch_input_ids.append(request.input_ids)
                batch_attention_masks.append(request.attention_mask)
            
            # Combine for batch processing
            batched_input_ids = torch.cat(batch_input_ids, dim=0)
            batched_attention_masks = torch.cat(batch_attention_masks, dim=0)
            
            # Forward pass for fresh requests
            with torch.no_grad():
                outputs = self.model(
                    input_ids=batched_input_ids,
                    attention_mask=batched_attention_masks,
                    use_cache=True
                )
            
            # Process results for each request
            for batch_idx, (request_id, request) in enumerate(request_groups["fresh_requests"]):
                with self.lock:
                    if request_id not in self.active_requests:
                        continue
                    
                    # Get logits for this request
                    next_token_logits = outputs.logits[batch_idx, -1, :]
                    
                    # Get full token sequence for repetition penalty
                    full_sequence = request.get_full_sequence()
                    
                    # Sample next token with repetition penalty
                    next_token = self._sample_token(
                        next_token_logits, 
                        request.temperature, 
                        request.top_p,
                        request.top_k,
                        request.repetition_penalty,
                        full_sequence
                    )
                    request.generated_ids.append(next_token)
                    
                    # Store KV cache
                    if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
                        # Extract this request's past_key_values
                        request_past_kv = tuple(
                            (layer_past[0][batch_idx:batch_idx+1], layer_past[1][batch_idx:batch_idx+1])
                            for layer_past in outputs.past_key_values
                        )
                        self.request_caches[request_id].update(request_past_kv)
                    
                    # Check if generation is complete (including EOS check)
                    if request.is_finished(self.tokenizer.eos_token_id):
                        request.done = True
        
        # Process continuation requests (with KV cache)
        for request_id, request in request_groups["continuation_requests"]:
            with self.lock:
                if request_id not in self.active_requests:
                    continue
                
                # Process with KV cache
                kv_cache = self.request_caches[request_id].get()
                last_token = torch.tensor([[request.generated_ids[-1]]], device=request.input_ids.device)
                
                # Forward pass with KV cache
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=last_token,
                        attention_mask=request.attention_mask,
                        past_key_values=kv_cache,
                        use_cache=True
                    )
                
                # Get full sequence for repetition penalty
                full_sequence = request.get_full_sequence()
                
                # Process results
                next_token_logits = outputs.logits[0, -1, :]
                next_token = self._sample_token(
                    next_token_logits, 
                    request.temperature, 
                    request.top_p,
                    request.top_k,
                    request.repetition_penalty,
                    full_sequence
                )
                request.generated_ids.append(next_token)
                
                # Update attention mask for the new token
                request.attention_mask = torch.cat([
                    request.attention_mask, 
                    torch.ones((1, 1), device=request.attention_mask.device)
                ], dim=1)
                
                # Update KV cache
                if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
                    self.request_caches[request_id].update(outputs.past_key_values)
                
                # Check if generation is complete (including EOS check)
                if request.is_finished(self.tokenizer.eos_token_id):
                    request.done = True
    
    def _sample_token(self, logits, temperature, top_p, top_k=0, repetition_penalty=1.0, prev_tokens=None):
        """Sample next token using temperature, top-p, top-k, and repetition penalties"""
        # Apply repetition penalty if provided
        if repetition_penalty != 1.0 and prev_tokens is not None and len(prev_tokens) > 0:
            # Convert to tensor for efficient indexing if it's a list
            if isinstance(prev_tokens, list):
                prev_tokens = torch.tensor(prev_tokens, device=logits.device)
            # Apply penalty - reduce probability of tokens that have already appeared
            for token_id in set(prev_tokens.tolist()):
                logits[token_id] /= repetition_penalty
        
        if temperature > 0:
            logits = logits / temperature
            
        # Apply top-k filtering if specified
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))  # Safety check
            values, _ = torch.topk(logits, top_k)
            min_value = values[-1]
            logits = torch.where(logits < min_value, 
                                torch.ones_like(logits) * -float('inf'),
                                logits)
            
        # Apply top-p filtering    
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[0] = False  # Keep at least the top token
            
            # Scatter sorted indices back to original indices
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(0, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('inf')
            
        # Sample from the filtered distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        
        return next_token
        
    def stop(self):
        """Stop the batcher thread"""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)


###################
# GENERATION FUNCTIONS
###################

def generate_text(model, tokenizer, prompt, max_tokens=50, temperature=1.0, top_p=0.9, top_k=0, 
                 use_kv_cache=True, repetition_penalty=1.0):
    """Generate text from a prompt using the model with KV cache optimization"""
    model.eval()
    
    # Determine what device the model is on
    if hasattr(model, 'device'):
        device = model.device
    else:
        # If model is spread across devices, use the first parameter's device
        device = next(model.parameters()).device
    
    # Encode the prompt
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device) if hasattr(encoded, 'attention_mask') else torch.ones_like(input_ids)
    
    # Time the generation
    start_time = time.time()
    generated_ids = []
    past_key_values = None
    
    with torch.no_grad():
        # Process the full prompt first
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_kv_cache
        )
        
        # Get next token logits
        next_token_logits = outputs.logits[0, -1, :]
        
        # Apply repetition penalty to input tokens
        if repetition_penalty > 1.0:
            input_token_ids = input_ids[0].tolist()
            # Penalize tokens that appeared in the prompt
            for token_id in set(input_token_ids):
                next_token_logits[token_id] /= repetition_penalty
        
        # Apply sampling (temperature, top-p, top-k)
        if temperature > 0:
            next_token_logits = next_token_logits / temperature
        
        # Apply top-k filtering if specified
        if top_k > 0:
            top_k = min(top_k, next_token_logits.size(-1))  # Safety check
            values, _ = torch.topk(next_token_logits, top_k)
            min_value = values[-1]
            next_token_logits = torch.where(next_token_logits < min_value,
                                   torch.ones_like(next_token_logits) * -float('inf'),
                                   next_token_logits)
        
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[0] = False  # Keep at least the top token
            
            # Scatter sorted indices to original indices
            indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool)
            indices_to_remove.scatter_(0, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = -float('inf')
        
        # Sample from the filtered distribution
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        generated_ids.append(next_token)
        
        # Check for EOS token
        if next_token == tokenizer.eos_token_id:
            pass  # We'll still include this EOS token but won't generate further
        
        # Update KV cache if using it
        if use_kv_cache:
            past_key_values = outputs.past_key_values
            
        # Generate remaining tokens auto-regressively
        for _ in range(max_tokens - 1):
            # Check if we've already hit EOS
            if generated_ids[-1] == tokenizer.eos_token_id:
                break
                
            # Prepare inputs for the next iteration
            new_input_ids = torch.tensor([[next_token]], device=device)
            
            # Extend attention mask for the new token
            if use_kv_cache:
                attention_mask = torch.cat([
                    attention_mask, 
                    torch.ones((1, 1), device=device, dtype=attention_mask.dtype)
                ], dim=1)
            
            # Forward pass
            outputs = model(
                input_ids=new_input_ids,
                attention_mask=attention_mask,  # Always provide attention mask
                past_key_values=past_key_values if use_kv_cache else None,
                use_cache=use_kv_cache
            )
            
            # Get logits and sample next token
            next_token_logits = outputs.logits[0, -1, :]
            
            # Apply repetition penalty on the full sequence
            if repetition_penalty > 1.0:
                full_sequence = input_ids[0].tolist() + generated_ids
                for token_id in set(full_sequence):
                    next_token_logits[token_id] /= repetition_penalty
            
            # Apply sampling (temperature, top-p, top-k)
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering if specified
            if top_k > 0:
                top_k = min(top_k, next_token_logits.size(-1))  # Safety check
                values, _ = torch.topk(next_token_logits, top_k)
                min_value = values[-1]
                next_token_logits = torch.where(next_token_logits < min_value,
                                    torch.ones_like(next_token_logits) * -float('inf'),
                                    next_token_logits)
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[0] = False  # Keep at least the top token
                
                # Scatter sorted indices to original indices
                indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool)
                indices_to_remove.scatter_(0, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('inf')
            
            # Sample from the filtered distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            generated_ids.append(next_token)
            
            # Check for EOS token
            if next_token == tokenizer.eos_token_id:
                break
            
            # Update KV cache
            if use_kv_cache:
                past_key_values = outputs.past_key_values
    
    # Combine input and generated tokens
    all_ids = input_ids[0].tolist() + generated_ids
    
    # Decode the generated text
    generated_text = tokenizer.decode(all_ids, skip_special_tokens=True)
    
    # Calculate generation statistics
    total_time = time.time() - start_time
    tokens_per_second = (len(generated_ids) + len(input_ids[0])) / total_time
    
    return generated_text, total_time, tokens_per_second


def generate_text_batch(model, tokenizer, prompts, max_tokens=50, temperature=1.0, top_p=0.9, top_k=0, repetition_penalty=1.0):
    """Generate text for multiple prompts using continuous batching"""
    # Determine what device the model is on
    if hasattr(model, 'device'):
        device = model.device
    else:
        # If model is spread across devices, use the first parameter's device
        device = next(model.parameters()).device
    
    # Create the continuous batcher
    batcher = ContinuousBatcher(model, tokenizer, max_batch_size=len(prompts))
    
    # Add all requests to the batcher
    start_time = time.time()
    
    for i, prompt in enumerate(prompts):
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded.input_ids.to(device)
        
        request = Request(
            input_ids, 
            max_tokens=max_tokens, 
            temperature=temperature, 
            top_p=top_p, 
            top_k=top_k,
            repetition_penalty=repetition_penalty
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
                    input_tokens = tokenizer(prompts[i], return_tensors="pt").input_ids[0].tolist()
                    
                    # Combine input and generated tokens
                    all_tokens = input_tokens + result
                    
                    # Decode the final text
                    results[i] = tokenizer.decode(all_tokens, skip_special_tokens=True)
                else:
                    all_done = False
        
        if not all_done:
            time.sleep(0.1)
    
    # Stop the batcher thread
    batcher.stop()
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Return results in the same order as prompts
    return [results[i] for i in range(len(prompts))], total_time


###################
# MAIN FUNCTIONS
###################

def demo(args):
    """Run a demonstration of the model's generation capabilities"""
    logger.info(f"Initializing on {args.device}...")
    
    # Load model and tokenizer with appropriate settings
    model, tokenizer = load_model_and_tokenizer(
        args.model_name, 
        args.device, 
        args.quantize,
        args.trust_remote_code,
        not args.no_opt_attention
    )
    
    # Use provided prompt or defaults
    if args.prompt:
        prompts = [args.prompt]
    else:
        prompts = [
            "The capital of China is",
            "The largest mammal on Earth is",
            "The theory of relativity states that"
        ]
    
    print(f"\n{'-'*50}")
    print(f"Generation parameters: max_tokens={args.max_tokens}, temperature={args.temperature}, "
          f"top_p={args.top_p}, top_k={args.top_k}, repetition_penalty={args.repetition_penalty}")
    print(f"Using {'batched' if args.batch else 'single'} inference with KV cache {'disabled' if args.no_kv_cache else 'enabled'}")
    print(f"{'-'*50}\n")
    
    if args.batch:
        # Use continuous batching for multiple prompts
        print(f"Generating completions for {len(prompts)} prompts in batch mode...")
        outputs, total_time = generate_text_batch(
            model, 
            tokenizer, 
            prompts, 
            max_tokens=args.max_tokens, 
            temperature=args.temperature, 
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty
        )
        
        total_tokens = 0
        for i, output in enumerate(outputs):
            prompt_tokens = len(tokenizer(prompts[i]).input_ids)
            total_tokens += prompt_tokens + args.max_tokens
            print(f"\nPrompt {i+1}: {prompts[i]}")
            print(f"Completion: {output}")
        
        print(f"\nGenerated {len(prompts)} completions in {total_time:.2f} seconds")
        print(f"Average time per completion: {total_time/len(prompts):.2f} seconds")
        print(f"Tokens per second: {total_tokens/total_time:.2f}")
    else:
        # Process prompts one at a time
        for i, prompt in enumerate(prompts):
            print(f"\nPrompt {i+1}: {prompt}")
            output, gen_time, tokens_per_second = generate_text(
                model, 
                tokenizer, 
                prompt, 
                max_tokens=args.max_tokens, 
                temperature=args.temperature, 
                top_p=args.top_p,
                top_k=args.top_k,
                use_kv_cache=not args.no_kv_cache,
                repetition_penalty=args.repetition_penalty
            )
            
            print(f"Completion: {output}")
            print(f"Generated in {gen_time:.2f} seconds")
            print(f"Tokens per second: {tokens_per_second:.2f}")


def main():
    """Parse arguments and run the demo"""
    parser = argparse.ArgumentParser(description='HuggingFace LLM Inference with Optimizations')
    
    # Model configuration
    parser.add_argument('--model_name', type=str, default='gpt2', 
                      help='Name of HuggingFace model to use (gpt2, Qwen/Qwen-7B-Chat, meta-llama/Llama-2-7b-chat-hf, etc.)')
    parser.add_argument('--trust_remote_code', action='store_true', 
                      help='Trust remote code (required for some models like Qwen)')
    
    # Generation settings
    parser.add_argument('--prompt', type=str, default='', help='Text prompt to complete')
    parser.add_argument('--max_tokens', type=int, default=50, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p (nucleus) sampling parameter')
    parser.add_argument('--top_k', type=int, default=0, help='Top-k sampling parameter (0 to disable)')
    parser.add_argument('--repetition_penalty', type=float, default=1.0, 
                      help='Penalty for repetition (1.0 = no penalty, >1.0 = reduce repetition)')
    
    # Inference optimizations
    parser.add_argument('--batch', action='store_true', help='Use continuous batching')
    parser.add_argument('--no_kv_cache', action='store_true', help='Disable KV cache')
    parser.add_argument('--half', action='store_true', help='Use half precision (float16)')
    parser.add_argument('--quantize', type=str, choices=['4bit', '8bit'], help='Use quantization (requires bitsandbytes)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'mps', 
                        help='Device to use (cuda/cpu/mps)')
    parser.add_argument('--no_opt_attention', action='store_true', 
                      help='Don\'t try to use optimized attention implementations')
    
    args = parser.parse_args()
    
    # Support Apple Silicon
    if args.device == 'mps' and torch.backends.mps.is_available():
        args.device = 'mps'
    elif args.device == 'cuda' and torch.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'
        logger.info("Falling back to CPU")
    
    # Run the demo
    demo(args)


if __name__ == "__main__":
    main()