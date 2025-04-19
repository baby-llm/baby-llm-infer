# engine.py
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import List, Dict, Optional
import time
from typing import List, Deque, Tuple
from sequence import Sequence, SequenceStatus
from scheduler import Scheduler, SchedulerConfig
from kv_cache import KVCacheManager, PastKeyValueType
from sampling import sample_logits
from utils import logger

class InferenceConfig:
    """Configuration for the generation process."""
    def __init__(self, temperature: float = 0.7, top_p: float = 0.9, top_k: int = 50):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

class InferenceEngine:
    """
    Drives the LLM inference process step-by-step, using the scheduler
    and KV cache manager.
    """
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, device: torch.device,
                 scheduler_config: SchedulerConfig, inference_config: InferenceConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.inference_config = inference_config
        self.cache_manager = KVCacheManager(model.config, device)
        self.scheduler = Scheduler(scheduler_config, self.cache_manager)
        self.eos_token_id = tokenizer.eos_token_id
        if self.eos_token_id is None:
             # Handle models like CodeLlama that might use a list
            if isinstance(tokenizer.eos_token_id, list):
                self.eos_token_id = tokenizer.eos_token_id[0]
                logger.warning(f"Using first EOS token ID: {self.eos_token_id}")
            else: # Fallback if truly missing
                 logger.warning("EOS token ID not found in tokenizer, generation might not stop correctly.")
                 # Assign a common value or handle error, here we'll just warn
                 self.eos_token_id = -1 # Indicate invalid

        logger.info("Inference Engine initialized.")

    def add_request(self, request_id: str, prompt: str, max_new_tokens: int):
        """Adds a new inference request."""
        prompt_token_ids = self.tokenizer.encode(prompt)
        seq = Sequence(request_id, prompt, prompt_token_ids, max_new_tokens)
        self.scheduler.add_sequence(seq)

    def _prepare_model_inputs(self, batch: List[Sequence]) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[PastKeyValueType]]]:
        """
        Prepares input tensors for the model's forward pass for the current step.
        Gathers the *last* token ID from each sequence in the batch.
        Retrieves the corresponding KV caches.
        Creates an attention mask.
        """
        input_ids_list: List[int] = []
        position_ids_list: List[int] = []
        past_key_values_list: List[Optional[PastKeyValueType]] = []
        seq_indices_with_cache: List[int] = [] # Track which sequences contribute to past_key_values

        max_len_in_batch = 0 # Needed for attention mask shape if using padding (not needed here)

        for i, seq in enumerate(batch):
            full_ids = seq.get_full_sequence_ids()
            last_token_id = full_ids[-1]
            input_ids_list.append(last_token_id)
            position_ids_list.append(seq.get_len() - 1) # Position of the last token
            max_len_in_batch = max(max_len_in_batch, seq.get_len())

            # Retrieve KV cache if it exists (i.e., not the first token)
            if seq.get_len() > len(seq.prompt_token_ids): # Or simply check if handle exists and cache is not None
                kv_cache = self.cache_manager.get_cache(seq.kv_cache_handle)
                if kv_cache is not None:
                    past_key_values_list.append(kv_cache)
                    seq_indices_with_cache.append(i) # Mark this sequence as having cache
                else:
                     # This case should ideally not happen if scheduling logic is correct
                     # after the first step, but handle defensively.
                     logger.warning(f"Seq {seq.request_id} expected KV cache but none found.")
                     past_key_values_list.append(None) # Indicate no cache for this one if needed by model format

            elif seq.get_len() == len(seq.prompt_token_ids): # Processing the very first token (prompt)
                # For the prompt processing step, past_key_values is None
                 past_key_values_list.append(None)


        # --- Input Tensor Creation ---
        # Shape: (batch_size, 1) - We only process one token per sequence per step
        input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long, device=self.device).unsqueeze(1)

        # --- Position IDs ---
        # Shape: (batch_size, 1)
        position_ids_tensor = torch.tensor(position_ids_list, dtype=torch.long, device=self.device).unsqueeze(1)


        # --- Attention Mask ---
        # In decoding, the attention mask allows attending to all previous tokens.
        # For processing just the *next* token, a simple mask might suffice,
        # or often models handle causal masking internally when past_key_values are provided.
        # Let's create a basic mask assuming causal attention is handled by the model or cache.
        # Shape: (batch_size, sequence_length) - where sequence_length is the MAX length in the batch *up to this point*
        # For single token decoding, shape (batch_size, 1) might seem right, but the model
        # needs to know the total context length via past_key_values or position_ids.
        # A common simplified approach for the N-th token (when cache is used):
        # Mask just needs to indicate the current position is valid.
        # The effective sequence length for attention is implicitly handled by the KV cache length + 1.
        # Let's use a simplified mask. The model internally combines this with cache length.
        # Shape needs to align with how the specific model expects it with past_key_values.
        # Often, just providing correct position_ids is sufficient for models like GPT-2 when using KV cache.
        # Let's try without a complex mask first, relying on position_ids and cache.
        # If needed, a proper causal mask up to max_len_in_batch could be built.
        # For simplicity, let's pass a basic mask of ones for the current token.
        attention_mask = torch.ones_like(input_ids_tensor, dtype=torch.long) # Shape: (batch_size, 1)

        # --- Reformat past_key_values ---
        # Hugging Face expects a tuple of tuples. If only some sequences have cache,
        # we might need to pad with None or handle differently depending on model.
        # This simplified version assumes we can pass the collected list if the model handles it,
        # OR more likely, we need to format it correctly.
        # Let's assume for now we run sequences WITH cache separately from sequences WITHOUT cache,
        # OR that the model call below handles mixed cache/no-cache if possible.
        # A robust implementation might need to structure the batch based on cache status.
        #
        # *Correction for HF*: `past_key_values` needs to be a tuple where each element corresponds
        # to a layer, and each layer's tuple contains tensors of shape [batch_size, num_heads, seq_len, head_dim].
        # We need to stack the individual sequence caches.

        formatted_pkv: Optional[PastKeyValueType] = None
        if any(pkv is not None for pkv in past_key_values_list):
            # We need to filter out Nones and stack correctly. Assume all running sequences
            # after the first token *have* a cache from the previous step.
            # Get caches only for sequences that are past the prompt phase.
            valid_caches = [self.cache_manager.get_cache(seq.kv_cache_handle)
                            for seq in batch if seq.get_len() > len(seq.prompt_token_ids) and seq.kv_cache_handle is not None]

            if valid_caches:
                # Stack tensors layer by layer
                num_layers = len(valid_caches[0])
                batch_size_with_cache = len(valid_caches)
                stacked_pkv = []
                for layer_idx in range(num_layers):
                    key_states = torch.cat([valid_caches[i][layer_idx][0] for i in range(batch_size_with_cache)], dim=0)
                    value_states = torch.cat([valid_caches[i][layer_idx][1] for i in range(batch_size_with_cache)], dim=0)
                    stacked_pkv.append((key_states, value_states))
                formatted_pkv = tuple(stacked_pkv)


        # We need to align inputs. Maybe run prompts and generation steps in separate micro-batches within step()?
        # Let's refine: Process all sequences in the batch. If it's the first token (prompt),
        # pass the full prompt. If it's a subsequent token, pass only the last token ID + cache.
        # This requires splitting the batch or more complex input prep.

        # --- *REVISED* Simpler Input Prep for MVP ---
        # Process only ONE token per sequence each step.
        # Assume the `batch` contains sequences ready for their *next* token generation.

        last_token_ids = [seq.get_full_sequence_ids()[-1] for seq in batch]
        input_ids_tensor = torch.tensor(last_token_ids, dtype=torch.long, device=self.device).unsqueeze(1) # Shape: (batch_size, 1)

        position_ids = [seq.get_len() - 1 for seq in batch]
        position_ids_tensor = torch.tensor(position_ids, dtype=torch.long, device=self.device).unsqueeze(1) # Shape: (batch_size, 1)

        # Attention mask: For single token decoding with KV cache, mask is often just [[1]] per sequence.
        # The model uses position_ids and cache length to know the context.
        attention_mask = torch.ones_like(input_ids_tensor) # Shape: (batch_size, 1)

        # Retrieve and stack KV caches for ALL sequences in the batch
        # (assuming they are all past the initial prompt processing which would happen separately or in first step)
        all_past_key_values = [self.cache_manager.get_cache(seq.kv_cache_handle) for seq in batch]

        # Filter out potential Nones (e.g., first step after prompt) and stack if caches exist
        valid_caches = [pkv for pkv in all_past_key_values if pkv is not None]
        formatted_pkv = None
        if valid_caches:
             # Check if all sequences in batch have cache or handle mixed case
             if len(valid_caches) == len(batch):
                num_layers = len(valid_caches[0])
                batch_size = len(batch)
                stacked_pkv = []
                for layer_idx in range(num_layers):
                    # Ensure tensors are on the correct device before concatenating
                    key_states = torch.cat([valid_caches[i][layer_idx][0].to(self.device) for i in range(batch_size)], dim=0)
                    value_states = torch.cat([valid_caches[i][layer_idx][1].to(self.device) for i in range(batch_size)], dim=0)
                    stacked_pkv.append((key_states, value_states))
                formatted_pkv = tuple(stacked_pkv)
             else:
                 # Handling mixed cache/no-cache requires more complex logic, potentially
                 # running them separately or padding. For MVP, assume batch homogeneity after prompt.
                 logger.error("Batch contains mix of sequences with/without KV cache - not supported in this simplified prep.")
                 # Fallback or raise error - return empty tensors for now
                 return torch.empty(0), torch.empty(0), None


        # Handle the initial prompt processing case separately (if needed) where formatted_pkv would be None
        if not valid_caches and any(seq.get_len() == len(seq.prompt_token_ids) for seq in batch):
             logger.debug("Processing initial prompt tokens, past_key_values is None.")
             # Ensure input_ids contains the full prompt for these sequences
             # This highlights the need for separate handling or more complex batch prep.
             # Let's assume this function is called *after* prompt processing for simplicity.
             pass # formatted_pkv remains None

        return input_ids_tensor, attention_mask, formatted_pkv, position_ids_tensor


    @torch.inference_mode()
    def step(self) -> List[Sequence]:
        """Performs one step of inference."""
        step_start_time = time.time()

        # 1. Get the batch for the current step from the scheduler
        active_batch = self.scheduler.schedule()
        if not active_batch:
            # logger.info("No active sequences to process.")
            return [] # Return empty list indicating nothing was processed

        # 2. Prepare inputs for the model
        #    This is tricky: prompt processing vs. single-token decoding.
        #    A real engine might run prompt processing separately first.
        #    For MVP: Assume first call processes prompts, subsequent calls process single tokens.

        is_prompt_batch = any(seq.get_len() == len(seq.prompt_token_ids) for seq in active_batch)

        if is_prompt_batch:
            # --- Handle Prompt Processing ---
            # This part needs careful batching if prompts have different lengths.
            # Using tokenizer padding.
            logger.debug(f"Processing prompts for batch size {len(active_batch)}")
            prompts = [seq.prompt_token_ids for seq in active_batch]
            # Pad prompts to the same length for batching
            padded_prompts = self.tokenizer.pad(
                {"input_ids": prompts},
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.model(
                input_ids=padded_prompts["input_ids"],
                attention_mask=padded_prompts["attention_mask"],
                past_key_values=None, # No cache for prompts
                use_cache=True # IMPORTANT: Get the initial KV cache
            )
            logits = outputs.logits
            past_key_values = outputs.past_key_values # Initial KV cache

            # Get logits for the *last* token of each prompt
            seq_lens = padded_prompts["attention_mask"].sum(dim=1)
            last_token_indices = seq_lens - 1
            batch_indices = torch.arange(len(active_batch), device=self.device)
            last_token_logits = logits[batch_indices, last_token_indices, :] # Shape: (batch_size, vocab_size)

            # Update cache for all sequences in the batch
            # The returned past_key_values corresponds to the whole batch.
            # We need to associate the correct part with each sequence handle.
            # This requires careful indexing or assuming the model returns it per-sequence implicitly in batch order.
            # Assuming batch order:
            for i, seq in enumerate(active_batch):
                 # Extract KV cache for this specific sequence (batch index i)
                 seq_pkv = tuple(
                     (layer_kv[0][i:i+1], layer_kv[1][i:i+1]) # Keep batch dim of 1
                     for layer_kv in past_key_values
                 )
                 self.cache_manager.update_cache(seq.kv_cache_handle, seq_pkv)

        else:
            # --- Handle Single Token Generation ---
            logger.debug(f"Generating next token for batch size {len(active_batch)}")
            input_ids, attention_mask, past_key_values, position_ids = self._prepare_model_inputs(active_batch)

            if input_ids.numel() == 0: # Handle error case from prep
                 logger.warning("Could not prepare valid inputs for generation step.")
                 return active_batch # Return batch as is, maybe mark as error later

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids, # Pass position IDs
                use_cache=True
            )
            # Logits shape: (batch_size, 1, vocab_size) - only for the new token
            logits = outputs.logits[:, -1, :] # Get logits for the last token position
            last_token_logits = logits # Shape: (batch_size, vocab_size)
            new_past_key_values = outputs.past_key_values

            # Update cache for all sequences
            for i, seq in enumerate(active_batch):
                 seq_pkv = tuple(
                     (layer_kv[0][i:i+1], layer_kv[1][i:i+1]) # Keep batch dim of 1
                     for layer_kv in new_past_key_values
                 )
                 self.cache_manager.update_cache(seq.kv_cache_handle, seq_pkv)


        # 3. Sample the next token for each sequence in the batch
        next_token_ids = sample_logits(
            last_token_logits,
            temperature=self.inference_config.temperature,
            top_p=self.inference_config.top_p,
            top_k=self.inference_config.top_k
        ) # Shape: (batch_size,)

        # 4. Update sequences and check for completion
        for i, seq in enumerate(active_batch):
            next_token = next_token_ids[i].item()
            seq.append_token_id(next_token)

            # Check stopping conditions
            if next_token == self.eos_token_id:
                logger.info(f"Sequence {seq.request_id} finished: EOS token.")
                seq.status = SequenceStatus.FINISHED_STOPPED
            elif len(seq.output_token_ids) >= seq.max_new_tokens:
                logger.info(f"Sequence {seq.request_id} finished: Max length.")
                seq.status = SequenceStatus.FINISHED_LENGTH

        step_end_time = time.time()
        logger.debug(f"Engine step finished in {step_end_time - step_start_time:.4f} seconds.")

        return active_batch # Return the processed batch

    def run_loop(self):
        """Runs the inference loop until all requests are processed."""
        logger.info("Starting inference loop...")
        loop_count = 0
        while self.scheduler.has_unfinished_requests():
            loop_count += 1
            logger.debug(f"\n--- Inference Loop Step {loop_count} ---")
            processed_batch = self.step()
            if not processed_batch and not self.scheduler.has_unfinished_requests():
                 # If step returned nothing AND scheduler is empty, we are done
                 break
            if not processed_batch and self.scheduler.has_unfinished_requests():
                 # If step returned nothing but scheduler still has requests (e.g., waiting)
                 # Avoid busy-waiting, maybe sleep briefly if purely waiting
                 time.sleep(0.01)
                 continue

            # Optional: Add a small delay to prevent potential busy-waiting if needed
            # time.sleep(0.001)
            if loop_count > 10000: # Safety break for potential infinite loops
                 logger.error("Inference loop reached max iterations. Exiting.")
                 break

        logger.info("Inference loop finished.")