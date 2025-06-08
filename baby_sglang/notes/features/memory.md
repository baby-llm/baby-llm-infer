# TODO 起一个合适的名称

**Scenario:** A user submits a prompt, and the LLM generates the subsequent text.

### 1. Initialization Phase (Model Loading)

* **Disk -> CPU RAM:** At program startup, the LLM's model weights (often tens to hundreds of GB) are loaded from disk into the main CPU memory (RAM).
* **CPU RAM -> GPU VRAM:** To leverage the GPU for high-speed parallel computation, the model weights are copied from CPU RAM to the GPU's video memory (VRAM). This process can be time-consuming. Once complete, the model is ready on the GPU to process requests.

### 2. Inference Phase - Prefill (Processing the Input Prompt)

* **User Input (CPU RAM):** The user's prompt text is first stored in CPU RAM.
* **Tokenization (CPU):** The CPU tokenizes the input text, converting it into a sequence of Token IDs.
* **Embedding Lookup (GPU):** The list of Token IDs is sent to the GPU. Using the embedding table stored in VRAM, the GPU converts the Token IDs into embedding vectors.
* **Attention Calculation (GPU):**
  * The GPU's Transformer layers process the embedding vectors of all input tokens in parallel.
  * In each attention layer, Query, Key, and Value vectors are computed for every token.
* **KV Cache Writing (GPU):** The computed Key and Value vectors (the KV cache) must be stored for use in generating subsequent tokens.
  * `TokenToKVPoolAllocator.alloc(need_size)`: Allocates storage slots (indices) on the GPU for these new KV pairs. It acquires the required number of free indices from the `free_slots` tensor, which also resides on the GPU.
  * `MHATokenToKVPool.set_kv_buffer(...)`: Writes the computed `cache_k` and `cache_v` tensors (on the GPU) into their corresponding physical memory locations in `k_buffer` and `v_buffer` on the GPU VRAM, guided by the slot indices allocated by the allocator.
* **Subsequent Layer Calculation (GPU):** The output vectors are passed through other layers, such as the Feed-Forward Network (FFN), to compute the probability distribution for the next token.

### 3. Inference Phase - Decoding (Generating New Tokens)

* **Token Selection (CPU/GPU):** The next token ID is sampled from the probability distribution generated in the previous step. This can be done on the GPU, but for more complex sampling strategies, the data might be sent to the CPU for processing.
* **Decoding Loop (Generating the N+1 Token):**
  * **Embedding Lookup (GPU):** The newly generated Token ID is sent to the GPU to look up its embedding vector.
  * **Attention Calculation (GPU):**
    * The GPU computes the Query vector for the new token.
    * **KV Cache Reading (GPU):** To calculate attention scores, the Key and Value vectors of all preceding tokens (from the prompt and previously generated tokens) are required. `MHATokenToKVPool.get_key_buffer(layer_id)` and `get_value_buffer(layer_id)` read this historical data for the current sequence directly from the KV cache in GPU VRAM.
    * **KV Cache Writing (GPU):** The Key and Value vectors for the new token are computed. `TokenToKVPoolAllocator.alloc(1)` allocates a single slot for this new KV pair, and `MHATokenToKVPool.set_kv_buffer(...)` writes the new `cache_k` and `cache_v` into it.
  * **Subsequent Layer Calculation (GPU):** Similar to the prefill phase, this step calculates the probability distribution for the next token (N+2).
* **Loop Continuation:** The decoding step is repeated until an end-of-sequence token is generated or the maximum length is reached.

### 4. KV Cache Management & Tiered Caching (GPU VRAM <-> CPU RAM)

* **VRAM Exhaustion:** As sequences grow, the KV cache (`k_buffer`, `v_buffer`) in GPU VRAM may become full.
* **Eviction (GPU -> CPU):** To free up space, "old" or "inactive" KV cache entries are moved (evicted) from GPU VRAM to CPU RAM.
  * A policy (e.g., LRU - Least Recently Used) selects KV pairs to evict.
  * `HostKVCache.alloc()`: Reserves space in CPU RAM.
  * `MHATokenToKVPool.get_flat_data(indices)`: Gathers the KV data to be evicted from GPU VRAM.
  * `MHATokenToKVPoolHost.transfer(indices, flat_data)`: Copies the data from the GPU to the `kv_buffer` in CPU RAM (often pinned memory for faster transfer).
  * `TokenToKVPoolAllocator.free(indices)`: Releases the GPU slot indices, making them available for reallocation.
* **Loading (CPU -> GPU):** If a computation requires KV data that was evicted (e.g., a swapped-out request becomes active), the data is moved back.
  * `TokenToKVPoolAllocator.alloc()`: Reallocates new slots on the GPU.
  * `MHATokenToKVPoolHost.get_flat_data(indices)`: Reads the KV data from CPU RAM.
  * `MHATokenToKVPool.transfer(indices, flat_data)`: Copies the data from CPU RAM back to the newly allocated slots in GPU VRAM.
* **Synchronization:** State management within `HostKVCache` (e.g., `MemoryStateInt`, `@synchronized` decorator) ensures data consistency and thread safety during transfers, preventing race conditions.

### 5. Completion Phase

* **Detokenization (CPU):** The final sequence of Token IDs is sent to the CPU and converted into human-readable text.
* **Resource Release:** Once a request is complete, all its KV cache slots (on both GPU and CPU) are freed using `TokenToKVPoolAllocator.free(indices)` and `HostKVCache.free(indices)`.

### Summary

The roles of the different memory and compute units are as follows:

* **Disk:** Persistent storage for the model.
* **CPU RAM:** Stores the model temporarily, handles text I/O, executes control logic, and acts as a secondary (host) cache for the KV cache when GPU VRAM is full.
* **GPU VRAM:** Stores the active model weights, holds the primary KV cache, and performs the heavy parallel computations (Embedding, Attention, FFN).
* **CPU:** Manages pre-processing (tokenization), post-processing (detokenization), scheduling, and data orchestration with the GPU.
* **GPU:** Handles the vast majority of compute-intensive tasks.

The system built upon `TokenToKVPoolAllocator`, `MHATokenToKVPool`, and `MHATokenToKVPoolHost` is a hierarchical KV cache. It prioritizes high-speed GPU VRAM and uses CPU RAM as a fallback, enabling support for longer sequences and higher concurrency. The performance of this system critically depends on efficiently managing data movement between VRAM and RAM.
