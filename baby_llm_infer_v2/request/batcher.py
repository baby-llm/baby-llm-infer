import time
import torch
from queue import Queue
from threading import Thread, Lock

from ..generation.sampling import sample_token
from .cache import KeyValueCache

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
                    next_token = sample_token(
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
                next_token = sample_token(
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
    
    def stop(self):
        """Stop the batcher thread"""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)