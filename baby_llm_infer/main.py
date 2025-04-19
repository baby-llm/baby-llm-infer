# main.py
import time
import uuid
from model_loader import load_model_and_tokenizer
from engine import InferenceEngine, InferenceConfig
from scheduler import SchedulerConfig
from utils import logger

def main():
    # --- Configuration ---
    # model_name = "gpt2" # Smallest standard GPT-2
    model_name = "distilgpt2" # Even smaller, faster for testing
    max_batch_size = 4 # Keep low for Colab/local testing
    max_new_tokens_per_request = 50 # Max tokens to generate per prompt

    scheduler_config = SchedulerConfig(max_batch_size=max_batch_size)
    inference_config = InferenceConfig(temperature=0.7, top_p=0.9, top_k=50)

    # --- Initialization ---
    model, tokenizer, device = load_model_and_tokenizer(model_name)
    engine = InferenceEngine(model, tokenizer, device, scheduler_config, inference_config) 

    # --- Request Handling ---
    requests = {} # Store request_id -> final generated text

    def add_new_request(prompt_text):
        request_id = str(uuid.uuid4())
        logger.info(f"Adding request {request_id}: '{prompt_text[:50]}...'")
        engine.add_request(
            request_id=request_id,
            prompt=prompt_text,
            max_new_tokens=max_new_tokens_per_request
        )
        requests[request_id] = {"prompt": prompt_text, "output": None, "status": "Waiting"}
        return request_id

    # --- Console Interaction ---
    print("\n--- MVP LLM Inference Framework ---")
    print(f"Model: {model_name}, Device: {device}, Max Batch Size: {max_batch_size}")
    print("Enter prompts below (type 'quit' or 'exit' to stop):")

    try:   
        while True:
            try:
                prompt = input("Enter prompt: ")
                if prompt.lower() in ["quit", "exit"]:
                    break
                if prompt.strip():
                    add_new_request(prompt)
                else:
                    print("Prompt cannot be empty.")

                # Run the engine until current requests finish (or for a few steps)
                # In a real server, the engine would run continuously in background.
                # Here, we run it explicitly after adding requests.
                print("Processing requests...")
                start_time = time.time()
                # Keep stepping until scheduler indicates no more work for now
                processed_in_cycle = 0
                while engine.scheduler.has_unfinished_requests():
                     engine.step() # Run one step
                     processed_in_cycle += 1
                     # Check for finished requests to update status (optional here, engine loop could do it)
                     # Safety break for this interactive demo
                     if processed_in_cycle > max_new_tokens_per_request * 2 + 10 : # Heuristic limit
                          logger.warning("Interactive cycle reached step limit.")
                          break
                     # Brief sleep to allow checking status without busy loop if needed
                     time.sleep(0.01)


                end_time = time.time()
                print(f"Processing took {end_time - start_time:.2f} seconds.")

                # Display results for finished requests
                print("\n--- Results ---")
                all_requests_done = True
                for req_id, data in requests.items():
                    # Find the sequence object in the engine's scheduler (or need a way to track finished)
                    # This part is a bit clunky in the demo - needs better state tracking
                    seq = engine.scheduler.running_pool.get(req_id)
                    # If not running, it might be finished or waiting. We need access to finished seqs.
                    # Let's assume for demo, if not running and not waiting, it finished.
                    # A better way: Engine should maintain a finished dict or return finished seqs.
                    status_str = "Unknown"
                    output_text = ""
                    if req_id in engine.scheduler.running_pool:
                        status_str = engine.scheduler.running_pool[req_id].status.name
                        all_requests_done = False
                    elif any(s.request_id == req_id for s in engine.scheduler.waiting_queue):
                         status_str = "WAITING"
                         all_requests_done = False
                    else: # Assume finished if not waiting or running
                         status_str = "FINISHED (assumed)"
                         # Need to retrieve final tokens - this is missing!
                         # Engine needs to store finished sequences' outputs temporarily.
                         # WORKAROUND for demo: Assume engine holds finished state internally
                         # In a real app, engine would emit results. We'll simulate later.
                         # For now, just show prompt. Output display needs engine enhancement.
                         # TODO: Enhance engine/main loop to store and retrieve final text.


                    print(f"Request {req_id[:8]}: Status={status_str}")
                    print(f"  Prompt: {data['prompt']}")
                    # print(f"  Output: {output_text}") # TODO: Retrieve final output
                    print("-" * 10)

                if all_requests_done and not engine.scheduler.has_unfinished_requests():
                    print("All requests processed.")
                    requests.clear() # Clear for next round in demo

            except EOFError: # Handle Ctrl+D
                break
            except KeyboardInterrupt: # Handle Ctrl+C
                print("\nExiting...")
                break

    finally:
        print("Framework shutting down.")
        # Clean up? (GPU memory usually released when process ends)

if __name__ == "__main__":
    main()