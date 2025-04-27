# mvp_inference_engine.py

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import sys

# --- Configuration ---
# Use a smaller model for easier running on various hardware
MODEL_NAME = "gpt2"  # ~500MB download, decent quality
# MODEL_NAME = "distilgpt2" # ~300MB download, faster, lower quality
# MODEL_NAME = "EleutherAI/pythia-70m" # Smaller Pythia model

# Set max new tokens to generate
MAX_NEW_TOKENS = 50

# --- Device Selection ---
def get_device():
    """Auto-selects the best available device."""
    if torch.cuda.is_available():
        print("Using CUDA (Nvidia GPU)")
        return torch.device("cuda")
    # Check for Apple Silicon MPS support
    # Need PyTorch 1.12+ (or 2.0+ recommended) and macOS 12.3+
    # M3 requires PyTorch 2.1+ nightly or 2.2+ stable for best support
    elif torch.backends.mps.is_available():
        # Check if MPS is functional
        try:
            # Simple test tensor calculation
            _ = torch.tensor([1], device="mps") * 2
            print("Using MPS (Apple Silicon GPU)")
            return torch.device("mps")
        except Exception as e:
            print(f"MPS device found but test failed: {e}")
            print("Falling back to CPU.")
            return torch.device("cpu")
    else:
        print("Using CPU")
        return torch.device("cpu")

DEVICE = get_device()

# --- Simple Inference Engine with KV Caching ---
class SimpleLLMEngine:
    def __init__(self, model_name: str, device: torch.device):
        """
        Initializes the model and tokenizer.
        """
        print(f"Loading model '{model_name}' to {device}...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Ensure tokenizer has a padding token, add if necessary
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
            print("Added EOS token as PAD token.")

        self.device = device
        self.model.eval() # Set model to evaluation mode
        print("Model and tokenizer loaded.")

    @torch.no_grad() # Disable gradient calculations for inference
    def generate(self, prompt: str, max_new_tokens: int):
        """
        Generates text based on the prompt using KV caching.
        Yields tokens one by one (streaming).
        """
        print("\n--- Starting Generation ---")
        print(f"Prompt: '{prompt}'")

        # 1. Tokenize the input prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        # attention_mask = inputs.attention_mask.to(self.device) # Needed if using padding

        # --- Core Inference Loop with KV Cache ---
        # `past_key_values` is the KV cache. It stores the computed Key and Value states
        # from previous attention layers. It starts empty.
        past_key_values = None
        generated_token_ids = []

        start_time = time.time()
        prompt_processing_finished = False

        for i in range(max_new_tokens):
            # Prepare model inputs
            if past_key_values is None:
                # First pass (Prompt Processing): Process the entire prompt
                current_input_ids = input_ids
                print(f"Step {i+1} (Prompt): Input IDs shape: {current_input_ids.shape}")
            else:
                # Subsequent passes (Token Generation): Process only the *last* generated token
                # This is the core optimization enabled by KV Caching!
                current_input_ids = new_token_id.unsqueeze(0) # Shape: (1, 1)
                if not prompt_processing_finished:
                    prompt_processing_time = time.time() - start_time
                    print(f"Prompt processing finished in {prompt_processing_time:.3f} s")
                    prompt_processing_finished = True
                print(f"Step {i+1} (Token): Input IDs shape: {current_input_ids.shape}")

            # 2. Forward Pass
            # Pass `past_key_values` to the model. If it's not None, the model
            # will reuse the cached states and only compute KVs for the new input token(s).
            # The model outputs the new `past_key_values` containing the updated cache.
            outputs = self.model(
                input_ids=current_input_ids,
                past_key_values=past_key_values,
                use_cache=True # This MUST be True to get the cache back
                # attention_mask=attention_mask # Needed if using padding or batching
            )

            # 3. Get Logits & Sample Next Token
            # We only need the logits for the *last* token in the sequence to predict the next one.
            next_token_logits = outputs.logits[:, -1, :] # Shape: (batch_size, vocab_size)
            # Simple greedy sampling (argmax)
            new_token_id = torch.argmax(next_token_logits, dim=-1) # Shape: (batch_size,)

            # 4. Store the Updated KV Cache
            # `outputs.past_key_values` contains the KVs for *all* tokens processed so far.
            # We store this to pass it into the *next* iteration.
            past_key_values = outputs.past_key_values
            # Debug: Print shape of the first layer's key cache tensor
            # It should grow in the sequence length dimension: (batch_size, num_heads, sequence_length, head_dim)
            if i == 0: # After first (prompt) pass
                 print(f"  KV Cache (Layer 0 Key) shape after prompt: {past_key_values[0][0].shape}")
            elif i == 1: # After first token generation
                 print(f"  KV Cache (Layer 0 Key) shape after 1st token: {past_key_values[0][0].shape}")


            # 5. Yield the Generated Token
            generated_token_ids.append(new_token_id.item())
            decoded_token = self.tokenizer.decode(new_token_id)
            # print(f"  Token {i+1}: ID={new_token_id.item()}, Decoded='{decoded_token}'")
            yield decoded_token

            # 6. Check for Stop Condition (EOS token)
            if new_token_id.item() == self.tokenizer.eos_token_id:
                print("\nEOS token generated. Stopping.")
                break

            # --- Update attention_mask for next iteration if needed ---
            # If using padding/batching, the attention mask needs to be extended
            # to include the new token. For this single-sequence example, it's
            # implicitly handled by the KV cache mechanism and input length.
            # Example: attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), device=self.device)], dim=1)

        end_time = time.time()
        total_time = end_time - start_time
        num_generated = len(generated_token_ids)
        tokens_per_sec = num_generated / (total_time - prompt_processing_time) if prompt_processing_finished and total_time > prompt_processing_time else 0

        print("\n--- Generation Complete ---")
        print(f"Generated {num_generated} tokens in {total_time:.3f} seconds.")
        if tokens_per_sec > 0:
            print(f"Generation speed (tokens/sec): {tokens_per_sec:.2f}")


# --- Main Execution ---
if __name__ == "__main__":
    # Instantiate the engine
    engine = SimpleLLMEngine(MODEL_NAME, DEVICE)

    # Define a prompt
    # prompt = "Alan Turing was a"
    prompt = "The capital of china is"

    # Generate and stream the output
    full_response = ""
    sys.stdout.write(f"Prompt: {prompt}") # Print prompt first
    sys.stdout.flush()

    for token in engine.generate(prompt, max_new_tokens=MAX_NEW_TOKENS):
        sys.stdout.write(token) # Print token as it arrives
        sys.stdout.flush()
        full_response += token

    print("\n--- Final Output ---")
    print(prompt + full_response)