import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import time
import logging
import sys

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Simple inference engine for standard LLMs")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model identifier to load from HuggingFace or local path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help="Device to run inference on",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling (lower = more deterministic)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top p for nucleus sampling",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="Top k for sampling (0 = disabled)",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty (1.0 = disabled)",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Disable KV cache for generation",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt to use for generation (if not provided, will ask for input)",
    )
    parser.add_argument(
        "--batch_demo",
        action="store_true",
        help="Run batch generation demo with predefined prompts",
    )
    return parser.parse_args()

def get_device(device_str):
    """Determine the appropriate device to use"""
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str

def load_model(args):
    """Load model and tokenizer"""
    device = get_device(args.device)
    logger.info(f"Initializing on {device}...")
    
    # Load model
    logger.info(f"Loading model {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model = model.to(device)
    
    # Log model size
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Loaded model with {num_params:,} trainable parameters")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Make sure padding token is set
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = "</s>"
    
    return model, tokenizer

def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
    top_k=0,
    repetition_penalty=1.0,
    use_kv_cache=True,
):
    """Generate text with the provided model and parameters"""
    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    if use_kv_cache:
        return generate_with_kv_cache(
            model, tokenizer, input_ids, 
            max_new_tokens, temperature, top_p, top_k, repetition_penalty
        )
    else:
        # Use the model's generate method
        start_time = time.time()
        
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
        }
        
        if temperature > 0:
            generation_config["temperature"] = temperature
            generation_config["top_p"] = top_p
            generation_config["do_sample"] = True
            if top_k > 0:
                generation_config["top_k"] = top_k
        else:
            generation_config["do_sample"] = False
        
        outputs = model.generate(input_ids, **generation_config)
        gen_time = time.time() - start_time
        
        # Decode the generated text
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Calculate tokens per second
        num_tokens_generated = outputs.shape[1] - input_ids.shape[1]
        tokens_per_second = num_tokens_generated / gen_time if gen_time > 0 else 0
        
        return output_text, gen_time, tokens_per_second

def generate_with_kv_cache(
    model, tokenizer, input_ids, 
    max_new_tokens, temperature, top_p, top_k, repetition_penalty
):
    """Generation with manual KV caching"""
    start_time = time.time()
    
    # Store generated tokens
    all_token_ids = input_ids[0].tolist()
    generated_tokens = []
    
    # Whether to sample or use greedy decoding
    do_sample = temperature > 0
    
    # First pass - process the entire prompt
    past_key_values = None
    
    for _ in range(max_new_tokens):
        # Forward pass
        with torch.no_grad():
            if past_key_values is None:
                # First pass - process the entire prompt
                outputs = model(
                    input_ids=input_ids,
                    use_cache=True
                )
                # Get the next token logits from the last position
                next_token_logits = outputs.logits[0, -1, :]
            else:
                # For subsequent passes, we need only the most recent token
                current_token = torch.tensor([[all_token_ids[-1]]], device=model.device)
                outputs = model(
                    input_ids=current_token,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                # Get the next token logits
                next_token_logits = outputs.logits[0, -1, :]
            
            # Update past_key_values for the next iteration
            past_key_values = outputs.past_key_values
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for prev_token in all_token_ids:
                    next_token_logits[prev_token] /= repetition_penalty
            
            # Sample from the distribution
            if do_sample:
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k = min(top_k, next_token_logits.size(-1))
                    values, _ = torch.topk(next_token_logits, top_k)
                    min_value = values[-1]
                    next_token_logits[next_token_logits < min_value] = -float('Inf')
                
                # Sample from the filtered distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits).item()
        
        # Add the token to our list
        all_token_ids.append(next_token)
        generated_tokens.append(next_token)
        
        # Check for EOS token
        if next_token == tokenizer.eos_token_id:
            break
    
    # Calculate generation metrics
    gen_time = time.time() - start_time
    
    # Decode the generated text
    output_text = tokenizer.decode(all_token_ids, skip_special_tokens=True)
    
    # Calculate tokens per second
    tokens_per_second = len(generated_tokens) / gen_time if gen_time > 0 else 0
    
    return output_text, gen_time, tokens_per_second

def batch_generate(model, tokenizer, prompts, args):
    """Generate outputs for multiple prompts"""
    results = []
    
    logger.info(f"Generating completions for {len(prompts)} prompts...")
    total_start_time = time.time()
    
    # Process each prompt
    for prompt in prompts:
        output, gen_time, tokens_per_second = generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            use_kv_cache=not args.no_cache,
        )
        results.append({
            "prompt": prompt,
            "completion": output,
            "time": gen_time,
            "tokens_per_second": tokens_per_second
        })
    
    total_time = time.time() - total_start_time
    avg_time = total_time / len(prompts)
    
    # Calculate average tokens per second
    total_tokens_per_second = sum(r["tokens_per_second"] for r in results) / len(results)
    
    logger.info(f"Generated {len(prompts)} completions in {total_time:.2f} seconds")
    logger.info(f"Average time per completion: {avg_time:.2f} seconds")
    logger.info(f"Tokens per second: {total_tokens_per_second:.2f}")
    
    return results, total_time, total_tokens_per_second

def interactive_demo(model, tokenizer, args):
    """Run an interactive demo where user enters prompts"""
    print("\n" + "=" * 50)
    print("Interactive mode - type a prompt and press Enter.")
    print("Enter 'q' to quit.")
    print("=" * 50 + "\n")
    
    while True:
        prompt = input("Prompt: ")
        if prompt.lower() == 'q':
            break
        
        print("\nGenerating...", flush=True)
        
        output, gen_time, tokens_per_second = generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            use_kv_cache=not args.no_cache,
        )
        
        print(f"\nOutput: {output}")
        print(f"Generation time: {gen_time:.2f} seconds")
        print(f"Tokens per second: {tokens_per_second:.2f}\n")

def demo(args):
    """Run the demo based on provided arguments"""
    model, tokenizer = load_model(args)
    
    # Batch demo with predetermined prompts
    if args.batch_demo:
        # Set up some simple test prompts
        prompts = [
            "The capital of France is",
            "The largest planet in our solar system is",
            "The theory of relativity states that"
        ]
        
        # Print generation parameters
        gen_params = {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "use_kv_cache": not args.no_cache,
        }
        
        print("\n" + "-" * 50)
        print(f"Generation parameters: {gen_params}")
        print("-" * 50 + "\n")
        
        # Generate for all prompts
        results, total_time, tokens_per_second = batch_generate(model, tokenizer, prompts, args)
        
        # Display results
        for i, result in enumerate(results, 1):
            print(f"\nPrompt {i}: {result['prompt']}")
            print(f"Completion: {result['completion']}")
        
        print(f"\nGenerated {len(results)} completions in {total_time:.2f} seconds")
        print(f"Average time per completion: {total_time/len(results):.2f} seconds")
        print(f"Tokens per second: {tokens_per_second:.2f}")
    
    # Single prompt provided via argument
    elif args.prompt:
        output, gen_time, tokens_per_second = generate_text(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            use_kv_cache=not args.no_cache,
        )
        
        print(f"\nPrompt: {args.prompt}")
        print(f"Output: {output}")
        print(f"Generation time: {gen_time:.2f} seconds")
        print(f"Tokens per second: {tokens_per_second:.2f}")
    
    # Interactive mode
    else:
        interactive_demo(model, tokenizer, args)
    
def main():
    args = parse_args()
    demo(args)

if __name__ == "__main__":
    main()