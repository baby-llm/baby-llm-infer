import time
import torch
from queue import Queue
from threading import Thread, Lock
from typing import Dict, Tuple, List, Optional, Any
import logging
import copy
import numpy as np

from .interfaces import Cache, GenerationRequest
from .cache import KeyValueCache
from .request import Request
from ..generation.sampling.strategies import TopPTopKSampler, GreedyTokenSampler

logger = logging.getLogger('optimized_inference')

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
        
        # Detect if we're using a Qwen model
        self.is_qwen_model = hasattr(model, 'config') and hasattr(model.config, 'model_type') and 'qwen' in model.config.model_type.lower()
        logger.info(f"Model type: {model.config.model_type if hasattr(model, 'config') and hasattr(model.config, 'model_type') else 'unknown'}")
        
        # For Qwen models, store complete generation states
        if self.is_qwen_model:
            self.qwen_generation_states = {}
        else:
            # Maintain a KV cache for each active request (only for non-Qwen models)
            self.request_caches = {}
        
        # Create samplers for token selection
        self.samplers = {
            'greedy': GreedyTokenSampler(),
            'sampling': TopPTopKSampler()
        }
        
        if self.is_qwen_model:
            logger.info("Qwen model detected, using special batching with left padding")
        else:
            logger.info(f"Using standard batching with padding_side={tokenizer.padding_side}")
        
        # Start processing thread
        self.thread = Thread(target=self._process_batches)
        self.thread.daemon = True
        self.thread.start()
        
    def add_request(self, request_id, request: Request):
        """Add a new request to be processed"""
        with self.lock:
            if not self.is_qwen_model:
                # Create a new KV cache for this request (for standard models)
                self.request_caches[request_id] = KeyValueCache()
            else:
                # For Qwen, we'll initialize generation state on first processing
                self.qwen_generation_states[request_id] = {
                    "initialized": False,
                    "model_inputs": None,
                }
        self.request_queue.put((request_id, request))
        
    def get_result(self, request_id) -> Optional[List[int]]:
        """Get results for a specific request if completed"""
        with self.lock:
            if request_id in self.active_requests and self.active_requests[request_id].done:
                request = self.active_requests[request_id]
                # Clean up
                if self.is_qwen_model and request_id in self.qwen_generation_states:
                    del self.qwen_generation_states[request_id]
                elif not self.is_qwen_model and request_id in self.request_caches:
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
            try:
                if self.is_qwen_model:
                    # For Qwen, process each request individually for proper cache handling
                    for request_id, request in requests_to_process:
                        self._process_single_qwen(request_id, request)
                else:
                    # For standard models, use batch processing
                    self._process_batch_standard(requests_to_process)
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                # Mark all requests in this batch as done to avoid deadlocks
                with self.lock:
                    for request_id, request in requests_to_process:
                        if request_id in self.active_requests:
                            self.active_requests[request_id].generated_ids.append(self.tokenizer.eos_token_id)
                            self.active_requests[request_id].done = True
    
    def _process_single_qwen(self, request_id, request):
        """Process a single Qwen request using the model's native utilities"""
        with self.lock:
            if request_id not in self.active_requests:
                return
                
            # Check if we've initialized this request
            if not self.qwen_generation_states[request_id]["initialized"]:
                # First time processing this request - encode the prompt
                # For Qwen models, directly prepare all inputs with the model's utilities
                encoded = self.tokenizer(request.prompt, return_tensors="pt")
                device = next(self.model.parameters()).device
                input_ids = encoded.input_ids.to(device)
                attention_mask = encoded.attention_mask.to(device)
                
                # Create a model inputs dictionary for incremental generation
                model_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "use_cache": True,
                }
                
                # Store the initial state
                self.qwen_generation_states[request_id]["model_inputs"] = model_inputs
                self.qwen_generation_states[request_id]["initialized"] = True
            else:
                # Get the previous model inputs
                model_inputs = self.qwen_generation_states[request_id]["model_inputs"]
            
            # Forward pass using the model's inputs
            with torch.no_grad():
                outputs = self.model(**model_inputs)
            
            # Sample next token
            next_token_logits = outputs.logits[0, -1, :]
            
            # Get full sequence for repetition penalty
            full_sequence = model_inputs["input_ids"][0].tolist()
            if hasattr(request, "generated_ids") and request.generated_ids:
                full_sequence.extend(request.generated_ids)
            
            # Select sampler based on temperature
            sampler = self._get_sampler_for_request(request)
            
            # Sample next token with repetition penalty
            next_token = sampler.sample(
                next_token_logits, 
                request.sampling_config,
                full_sequence
            )
            request.add_token(next_token)
            
            # Update model inputs for next iteration using the model's prepare_inputs_for_generation
            next_token_tensor = torch.tensor([[next_token]], device=model_inputs["input_ids"].device)
            
            # Proper way to update inputs for Qwen model
            if hasattr(outputs, "past_key_values"):
                # Update using the model's own preparation logic
                updated_inputs = {
                    "input_ids": next_token_tensor,
                    "past_key_values": outputs.past_key_values,
                    "attention_mask": torch.cat([
                        model_inputs["attention_mask"],
                        torch.ones((1, 1), device=model_inputs["attention_mask"].device)
                    ], dim=1),
                    "use_cache": True,
                }
                
                # Store updated inputs for next iteration
                self.qwen_generation_states[request_id]["model_inputs"] = updated_inputs
            
            # Check if generation is complete
            if request.is_finished(self.tokenizer.eos_token_id):
                request.done = True
                
    def _process_batch_standard(self, requests):
        """Process a batch of requests for standard models"""
        # Prepare and categorize batch inputs
        request_groups = self._prepare_standard_model_inputs(requests)
        
        # Process fresh requests (no KV cache) in a batch
        if request_groups["fresh_requests"]:
            self._process_fresh_requests_standard(request_groups["fresh_requests"])
        
        # Process continuation requests (with KV cache)
        if request_groups["continuation_requests"]:
            for request_id, request in request_groups["continuation_requests"]:
                self._process_single_continuation_standard(request_id, request)
                
    def _prepare_standard_model_inputs(self, requests_to_process):
        """Prepare unified inputs for standard model forward pass"""
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
    
    def _process_fresh_requests_standard(self, fresh_requests):
        """Process fresh requests for standard models"""
        # Find the maximum sequence length in this batch
        max_length = max(request.input_ids.size(1) for _, request in fresh_requests)
        
        batch_input_ids = []
        batch_attention_masks = []
        orig_lengths = []  # Track original lengths for proper logit extraction
        
        for request_id, request in fresh_requests:
            # Get current dimensions
            current_length = request.input_ids.size(1)
            orig_lengths.append(current_length)
            
            # If we need to pad
            if current_length < max_length:
                # Pad input_ids with the padding token
                padding_size = max_length - current_length
                pad_tensor = torch.full(
                    (1, padding_size), 
                    self.tokenizer.pad_token_id, 
                    dtype=request.input_ids.dtype,
                    device=request.input_ids.device
                )
                
                # Apply padding based on tokenizer's padding_side
                if self.tokenizer.padding_side == "right":
                    padded_input_ids = torch.cat([request.input_ids, pad_tensor], dim=1)
                    # For right padding, attention mask is 1s for original tokens, 0s for padding
                    pad_mask = torch.zeros(
                        (1, padding_size),
                        dtype=request.attention_mask.dtype,
                        device=request.attention_mask.device
                    )
                    padded_attention_mask = torch.cat([request.attention_mask, pad_mask], dim=1)
                else:  # left padding
                    padded_input_ids = torch.cat([pad_tensor, request.input_ids], dim=1)
                    # For left padding, attention mask is 0s for padding, 1s for original tokens
                    pad_mask = torch.zeros(
                        (1, padding_size),
                        dtype=request.attention_mask.dtype,
                        device=request.attention_mask.device
                    )
                    padded_attention_mask = torch.cat([pad_mask, request.attention_mask], dim=1)
                
                batch_input_ids.append(padded_input_ids)
                batch_attention_masks.append(padded_attention_mask)
            else:
                # No padding needed
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
        for batch_idx, ((request_id, request), orig_length) in enumerate(zip(fresh_requests, orig_lengths)):
            with self.lock:
                if request_id not in self.active_requests:
                    continue
                
                # Get logits for this request - use the original sequence's last position
                if self.tokenizer.padding_side == "right":
                    # For right padding, the last token is at the original length - 1
                    logit_position = orig_length - 1
                else:
                    # For left padding, the last token is at the end regardless of padding
                    logit_position = batched_input_ids.size(1) - 1
                
                next_token_logits = outputs.logits[batch_idx, logit_position, :]
                
                # Get full token sequence for repetition penalty
                if self.tokenizer.padding_side == "right":
                    # For right padding, just use the original sequence without padding
                    full_sequence = batched_input_ids[batch_idx, :orig_length].tolist()
                else:
                    # For left padding, we need to skip padding tokens
                    padding_offset = batched_input_ids.size(1) - orig_length
                    full_sequence = batched_input_ids[batch_idx, padding_offset:].tolist()
                
                # Select sampler based on temperature
                sampler = self._get_sampler_for_request(request)
                
                # Sample next token with repetition penalty
                next_token = sampler.sample(
                    next_token_logits, 
                    request.sampling_config,
                    full_sequence
                )
                request.add_token(next_token)
                
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
    
    def _process_single_continuation_standard(self, request_id, request):
        """Process a single continuation for standard models"""
        with self.lock:
            if request_id not in self.active_requests:
                return
            
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
            
            # Select sampler based on temperature
            sampler = self._get_sampler_for_request(request)
            
            # Process results
            next_token_logits = outputs.logits[0, -1, :]
            next_token = sampler.sample(
                next_token_logits, 
                request.sampling_config,
                full_sequence
            )
            request.add_token(next_token)
            
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
    
    def _get_sampler_for_request(self, request):
        """Get the appropriate sampler based on request parameters"""
        if request.sampling_config.temperature == 0:
            return self.samplers['greedy']
        else:
            return self.samplers['sampling']
    
    def stop(self):
        """Stop the batcher thread"""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)