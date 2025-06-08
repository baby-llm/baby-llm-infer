from baby_sglang.snippet.engine import Engine
from baby_sglang.snippet.server_args import ServerArgs # Assuming ServerArgs is accessible

if __name__ == "__main__":
    # Define minimal ServerArgs or pass kwargs directly to Engine
    # These are some common arguments. You'll need to ensure
    # 'model_path' and 'tokenizer_path' point to something, even if dummy,
    # as they are often checked. 'model_max_len' is used by our placeholder.
    engine_args = {
        # "model_path": "dummy_model_path",  # Replace with a valid path if needed by other parts
        # "tokenizer_path": "dummy_tokenizer_path", # Or same as model_path if tokenizer is there
        # "model_mode": "dummy", # This might be needed by ServerArgs or other logic
        # "trust_remote_code": True, # Often needed for Hugging Face models
        # "tokenizer_mode": "auto",
        # "log_level": "info", # To see logs from processes
        # "host": "127.0.0.1",
        "port": 30000, # Main server port, not the internal ZMQ ports
        # "tp_size": 1, # Tensor parallelism size, must be 1 for your simplified setup
        # # --- Crucial for the placeholder scheduler ---
        # "model_max_len": 2048, # Example value, used by the placeholder scheduler
        # # Add any other ServerArgs your setup requires
    }

    print("Initializing Engine...")
    # You can pass kwargs which will be used to create ServerArgs internally
    engine = Engine(**engine_args)
    print("Engine initialized.")

    # Example prompt
    prompt_text = "Once upon a time"
    sampling_params_dict = {"temperature": 0.7, "max_new_tokens": 50}

    # Test non-streaming generation
    print("\n--- Non-streaming generation ---")
    try:
        output = engine.generate(
            prompt=prompt_text,
            sampling_params=sampling_params_dict,
            stream=False
        )
        print(f"Output (non-streaming): {output}")
        # Note: With placeholder functions, 'output' will likely be empty or just reflect
        #       the initial state because no actual token generation or detokenization happens.
        #       The key is that the call doesn't crash due to process setup issues.
    except Exception as e:
        print(f"Error during non-streaming generation: {e}")

    # Test streaming generation
    print("\n--- Streaming generation ---")
    try:
        stream_generator = engine.generate(
            prompt=prompt_text,
            sampling_params=sampling_params_dict,
            stream=True
        )
        print("Output (streaming):")
        for chunk in stream_generator:
            print(chunk, end="", flush=True)
            # Again, chunks will be minimal/empty with placeholders
        print("\nStream finished.")
    except Exception as e:
        print(f"Error during streaming generation: {e}")

    print("\nEngine tests complete. Subprocesses should be running in the background.")
    print("You might need to manually terminate this script (Ctrl+C) to stop subprocesses if they don't exit cleanly on their own yet.")
    # In a real application, you'd have a shutdown mechanism for the engine.
    # For now, the subprocesses (scheduler, detokenizer) will run their `while True` loop.
    # `atexit` is commented out in your current Engine, so manual stop might be needed.