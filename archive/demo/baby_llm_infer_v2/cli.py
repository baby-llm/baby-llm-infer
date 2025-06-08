import argparse
import torch
import logging
import json
from typing import Dict, Any, List, Optional

from .utils import setup_logger, get_optimal_device
from .generation.model import ModelFactory, TokenizerLoader
from .generation.engine import GenerationEngine
from .config.model_config import ModelConfig, QuantizationConfig, AttentionConfig
from .config.generation_config import GenerationConfig, SamplingConfig

logger = logging.getLogger('optimized_inference')

def create_configs_from_args(args) -> tuple:
    """Create model and generation configs from command-line arguments"""
    # Create model config
    quantization_config = QuantizationConfig(
        method=args.quantize if args.quantize else "none",
    )
    
    attention_config = AttentionConfig(
        use_optimized=not args.no_opt_attention
    )
    
    model_config = ModelConfig(
        model_name=args.model_name,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        torch_dtype="float16" if args.half else "auto",
        quantization=quantization_config,
        attention=attention_config
    )
    
    # Create generation config
    sampling_config = SamplingConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty
    )
    
    generation_config = GenerationConfig(
        max_tokens=args.max_tokens,
        sampling=sampling_config,
        use_kv_cache=not args.no_kv_cache
    )
    
    return model_config, generation_config

def demo(args):
    """Run a demonstration of the model's generation capabilities"""
    # Create configurations
    model_config, generation_config = create_configs_from_args(args)
    
    logger.info(f"Initializing on {model_config.device}...")
    
    # Load model and tokenizer
    try:
        model = ModelFactory.create_model(model_config)
        tokenizer = TokenizerLoader.load_tokenizer(
            model_config.model_name,
            model_config.trust_remote_code
        )
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        return
    
    # Create generation engine
    engine = GenerationEngine(model, tokenizer)
    
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
    print(f"Generation parameters: {json.dumps(generation_config.__dict__, default=lambda o: o.__dict__)}")
    print(f"Using {'batched' if args.batch else 'single'} inference with KV cache {'disabled' if args.no_kv_cache else 'enabled'}")
    print(f"{'-'*50}\n")
    
    try:
        if args.batch:
            # Use continuous batching for multiple prompts
            print(f"Generating completions for {len(prompts)} prompts in batch mode...")
            results = engine.generate_batch(prompts, generation_config)
            
            total_tokens = 0
            for i, result in enumerate(results):
                total_tokens += result["metrics"]["total_tokens"]
                print(f"\nPrompt {i+1}: {prompts[i]}")
                print(f"Completion: {result['text']}")
            
            # Get batch timing from the first result
            generation_time = results[0]["metrics"]["generation_time"]
            
            print(f"\nGenerated {len(prompts)} completions in {generation_time:.2f} seconds")
            print(f"Average time per completion: {generation_time/len(prompts):.2f} seconds")
            print(f"Tokens per second: {total_tokens/generation_time:.2f}")
        else:
            # Process prompts one at a time
            for i, prompt in enumerate(prompts):
                print(f"\nPrompt {i+1}: {prompt}")
                result = engine.generate(prompt, generation_config)
                
                print(f"Completion: {result['text']}")
                print(f"Generated in {result['metrics']['generation_time']:.2f} seconds")
                print(f"Tokens per second: {result['metrics']['tokens_per_second']:.2f}")
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")

def create_arg_parser():
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(description='HuggingFace LLM Inference with Optimizations')
    
    # Model configuration
    parser.add_argument('--model_name', type=str, default='gpt2', 
                      help='Name of HuggingFace model to use (gpt2, Qwen/Qwen-7B-Chat, meta-llama/Llama-2-7b-chat-hf, etc.)')
    parser.add_argument('--trust_remote_code', action='store_true', 
                      help='Trust remote code (required for some models like Qwen)')
    
    # Generation settings
    parser.add_argument('--prompt', type=str, default='', help='Text prompt to complete')
    parser.add_argument('--max_tokens', type=int, default=50, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature (0 for greedy)')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p (nucleus) sampling parameter')
    parser.add_argument('--top_k', type=int, default=0, help='Top-k sampling parameter (0 to disable)')
    parser.add_argument('--repetition_penalty', type=float, default=1.0, 
                      help='Penalty for repetition (1.0 = no penalty, >1.0 = reduce repetition)')
    
    # Inference optimizations
    parser.add_argument('--batch', action='store_true', help='Use continuous batching')
    parser.add_argument('--no_kv_cache', action='store_true', help='Disable KV cache')
    parser.add_argument('--half', action='store_true', help='Use half precision (float16)')
    parser.add_argument('--quantize', type=str, choices=['4bit', '8bit'], help='Use quantization (requires bitsandbytes)')
    parser.add_argument('--device', type=str, default='auto', 
                      help='Device to use (auto, cuda, cpu, mps)')
    parser.add_argument('--no_opt_attention', action='store_true', 
                      help='Don\'t try to use optimized attention implementations')
    
    return parser