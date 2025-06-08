from baby_sglang.sgl import server_args

"""
The entry point of inference server. (SRT = SGLang Runtime)

This file implements python APIs for the inference engine.
"""

class Engine(EngineBase):
    """
    The entry point to the inference engine.

    - The engine consists of three components:
        1. TokenizerManager: Tokenizes the requests and sends them to the scheduler.
        2. Scheduler (subprocess): Receives requests from the Tokenizer Manager, schedules batches, forwards them, and sends the output tokens to the Detokenizer Manager.
        3. DetokenizerManager (subprocess): Detokenizes the output tokens and sends the result back to the Tokenizer Manager.

    Note:
    1. The HTTP server, Engine, and TokenizerManager both run in the main process.
    2. Inter-process communication is done through ICP (each process uses a different port) via the ZMQ library.
    """

    def __init__(self, **kwargs):
        """
        The arguments of this function is the same as `sglang/srt/server_args.py::ServerArgs`.
        Please refer to `ServerArgs` for the documentation.
        """

        """ ============================== BabySGL =======================================
        Pseudocode
        1. Initialize Engine
           Read the parameters required by server_args from kwargs.
        2. Initialize other components including TokenizerManager, Scheduler, and DetokenizerManager.
           - Initialize TokenizerManager in the main process.
           - Initialize Scheduler and DetokenizerManager in subprocesses.
           - Establish one-way communication between components via ZMQ.

        Questions
        1. Why initialize Scheduler and DetokenizerManager in subprocesses? In other words,
           what resources are shared between TokenizerManager, Scheduler, and DetokenizerManager,
           and what resources are exclusive to each?
        
           Answer: This is done to overcome Python's Global Interpreter Lock (GIL), which
           prevents true parallelism in a single process. By moving the CPU-intensive tasks
           (Scheduling, Detokenization) into separate processes, they can run on different
           CPU cores concurrently without blocking the main process.
        
           - Exclusive Resources: Each process has its own memory and Python interpreter.
             - TokenizerManager: Exclusively owns the tokenizer model and its logic (CPU-bound).
             - Scheduler Process: Exclusively owns the LLM model weights on the GPU, manages
               the KV cache, and runs inference (GPU-bound).
             - DetokenizerManager: Exclusively owns the detokenization logic (CPU-bound).
           - Shared Resources: They do not share memory directly. They "share" communication
             channels (ZMQ sockets) through which they pass messages (requests and results).
        
        2. What is IPC, what is ZMQ in Python, and what is NCCL?
        
           Answer:
           - IPC (Inter-Process Communication): The general mechanism that allows separate
             processes, which have their own isolated memory, to communicate with each other.
        
           - ZMQ (ZeroMQ): A high-performance messaging library used for IPC. It acts like a
             "post office" allowing SGLang's different processes (Tokenizer, Scheduler, Detokenizer)
             to send messages to each other efficiently. It can use `ipc://` for fast, same-machine
             communication or `tcp://` for network communication.
        
           - NCCL (NVIDIA Collective Communications Library): A specialized, high-speed library for
             direct GPU-to-GPU communication. It's used when a model is split across multiple
             GPUs (`tp_size > 1`). NCCL bypasses the CPU to provide the highest possible
             bandwidth for the GPUs to exchange data during inference.
        """
        if "server_args" in kwargs:
            # Directly load server_args
            server_args = kwargs["server_args"]
        else:
            # Construct server_args from kwargs
            if "log_level" not in kwargs:
                # Do not print logs by default
                kwargs["log_level"] = "error"
            server_args = ServerArgs(**kwargs)

        # Shutdown the subprocesses automatically when the program exits
        atexit.register(self.shutdown)

        # Allocate ports for inter-process communications
        port_args = PortArgs.init_new(server_args)
        logger.info(f"{server_args=}")

        # Launch subprocesses
        tokenizer_manager, scheduler_info = _launch_subprocesses(
            server_args=server_args,
            port_args=port_args,
        )

        self.server_args = server_args
        self.tokenizer_manager = tokenizer_manager
        self.scheduler_info = scheduler_info

        context = zmq.Context(2)
        self.send_to_rpc = get_zmq_socket(
            context, zmq.DEALER, port_args.rpc_ipc_name, True
        )

    def generate(
        self,
        # The input prompt. It can be a single prompt or a batch of prompts.
        prompt: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Union[List[Dict], Dict]] = None,
        # The token ids for text; one can either specify text or input_ids.
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        # The image input. It can be an image instance, file name, URL, or base64 encoded string.
        # Can be formatted as:
        # - Single image for a single request
        # - List of images (one per request in a batch)
        # - List of lists of images (multiple images per request)
        # See also python/sglang/srt/utils.py:load_image for more details.
        image_data: Optional[
            Union[
                List[List[Union[Image, str]]],
                List[Union[Image, str]],
                Union[Image, str],
            ]
        ] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None,
        lora_path: Optional[List[Optional[str]]] = None,
        custom_logit_processor: Optional[Union[List[str], str]] = None,
        return_hidden_states: bool = False,
        stream: bool = False,
    ) -> Union[Dict, Iterator[Dict]]:
        """
        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::GenerateReqInput`.
        Please refer to `GenerateReqInput` for the documentation.
        """

        """ ============================== BabySGL ==============================
        Pseudocode

        """
        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
            lora_path=lora_path,
            custom_logit_processor=custom_logit_processor,
            return_hidden_states=return_hidden_states,
            stream=stream,
        )
        loop = asyncio.get_event_loop()
        generator = self.tokenizer_manager.generate_request(obj, None)

        if stream:

            def generator_wrapper():
                while True:
                    try:
                        chunk = loop.run_until_complete(generator.__anext__())
                        yield chunk
                    except StopAsyncIteration:
                        break

            return generator_wrapper()
        else:
            ret = loop.run_until_complete(generator.__anext__())
            return ret
        
    async def async_generate(
        self,
        # The input prompt. It can be a single prompt or a batch of prompts.
        prompt: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Union[List[Dict], Dict]] = None,
        # The token ids for text; one can either specify text or input_ids.
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        # The image input. It can be an image instance, file name, URL, or base64 encoded string.
        # Can be formatted as:
        # - Single image for a single request
        # - List of images (one per request in a batch)
        # - List of lists of images (multiple images per request)
        # See also python/sglang/srt/utils.py:load_image for more details.
        image_data: Optional[
            Union[
                List[List[Union[Image, str]]],
                List[Union[Image, str]],
                Union[Image, str],
            ]
        ] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None,
        lora_path: Optional[List[Optional[str]]] = None,
        custom_logit_processor: Optional[Union[List[str], str]] = None,
        stream: bool = False,
    ) -> Union[Dict, AsyncIterator[Dict]]:
        """
        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::GenerateReqInput`.
        Please refer to `GenerateReqInput` for the documentation.
        """
        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
            lora_path=lora_path,
            stream=stream,
            custom_logit_processor=custom_logit_processor,
        )
        generator = self.tokenizer_manager.generate_request(obj, None)

        if stream is True:
            return generator
        else:
            return await generator.__anext__()
        
    def shutdown(self):
        """Shutdown the engine"""
        kill_process_tree(os.getpid(), include_parent=False)


def _launch_subprocesses(
    server_args: ServerArgs, port_args: Optional[PortArgs] = None
) -> Tuple[TokenizerManager, Dict]:
    """
    Launch the TokenizerManager in the main process, the Scheduler in a subprocess, and the DetokenizerManager in another subprocess.
    """

    """ ============================== BabySGL =======================================
    # Question 1:
    # What do nnodes, node_rank, dp_size, dp_rank, tp_size, tp_rank, and gpu_id mean?
    # On which nodes do the Engine, TokenizerManager, Scheduler, and DetokenizerManager run?
    # How do multiple nodes communicate? How do multiple GPUs within a node communicate?
    # What is the path of a request through these nodes?
    # What concepts from distributed systems do these correspond to? Please provide a
    # systematic and comprehensive answer.
    
    Answer:
    
    ### Part 1: Definitions
    
    These parameters control the distributed setup for serving the LLM.
    
    *   `nnodes` (number of nodes): The total number of physical machines (or virtual machines) in your cluster.
    *   `node_rank`: The unique ID of the current machine, from `0` to `nnodes - 1`. The machine with `node_rank=0` is the "head node" or "master node," which is the entry point for all requests.
    *   `tp_size` (Tensor Parallelism size): The total number of GPUs across all nodes that a *single* model is split onto. This is used when a model is too large to fit on one GPU. For example, if `tp_size=8`, one copy of the model is spread across 8 GPUs.
    *   `tp_rank` (Tensor Parallelism rank): The unique ID of a GPU within the tensor-parallel group, from `0` to `tp_size - 1`.
    *   `dp_size` (Data Parallelism size): The number of identical copies (replicas) of the model that exist. Each replica can process different requests simultaneously, increasing throughput. If `tp_size=8` and `dp_size=2`, you have two identical 8-GPU models, using 16 GPUs in total.
    *   `dp_rank` (Data Parallelism rank): The unique ID of a model replica, from `0` to `dp_size - 1`.
    *   `gpu_id`: The physical device ID of a GPU on a specific node, as seen by CUDA (e.g., 0, 1, 2...).

    ### Part 2: Component Placement
    
    *   **Engine, TokenizerManager, DetokenizerManager**: These components run **only on the head node** (`node_rank=0`). They are the centralized I/O hub. The `Engine` receives user requests, the `TokenizerManager` preprocesses them, and the `DetokenizerManager` post-processes the results.
    *   **Scheduler**: A `Scheduler` process is created for **each GPU** involved in the computation. These processes run across all nodes. For a `tp_size=8`, `nnodes=2` setup, there will be 4 `Scheduler` processes on `node_rank=0` and 4 on `node_rank=1`. The `Scheduler` is responsible for managing the GPU and running the actual model inference.

    ### Part 3: Communication
    
    *   **Inter-Node Communication (Between Machines)**: This happens over the network.
        - **For control signals** (e.g., sending a tokenized request from the TokenizerManager to a remote Scheduler), SGLang uses **ZMQ over TCP** (`tcp://...`).
        - **For GPU-to-GPU data exchange** during inference (the core of Tensor Parallelism), it uses **NCCL**, which is highly optimized for transferring tensor data over network interfaces.
    
    *   **Intra-Node Communication (Within a Machine)**:
        - **For control signals**, SGLang uses **ZMQ over IPC** (`ipc://...`). This uses special files on the filesystem and is much faster than TCP for processes on the same machine.
        - **For GPU-to-GPU data exchange**, it also uses **NCCL**, but over high-speed interconnects like NVLink or PCIe, which is extremely fast.

    ### Part 4: Request Path (Simplified for TP)
    
    1.  A user's HTTP request arrives at the `Engine` on the **head node (rank 0)**.
    2.  The `Engine` passes it to the `TokenizerManager` (still on node 0).
    3.  The `TokenizerManager` converts the text to token IDs and sends this tokenized request via ZMQ to all `Scheduler` processes in the TP group, which are distributed across **all nodes**.
    4.  Each `Scheduler` process running on its assigned GPU performs its part of the model computation. The GPUs continuously and rapidly exchange data with each other using **NCCL**.
    5.  The final output tokens are generated and sent from the `Scheduler`s back to the `DetokenizerManager` on **node 0** via ZMQ.
    6.  The `DetokenizerManager` converts the tokens back into a text string and sends it to the `TokenizerManager` (which manages the request state).
    7.  The final result is returned to the user.
    
    ### Part 5: Corresponding Distributed Systems Concepts
    
    *   **Leader/Worker Pattern**: The head node (`node_rank=0`) acts as the leader, handling all external I/O and coordinating the work. The other nodes (`node_rank > 0`) are purely workers.
    *   **Model Parallelism**: This is exactly what Tensor Parallelism (`tp_size`) is—a form of model parallelism where a single large task is split across multiple workers.
    *   **Data Parallelism**: This corresponds directly to `dp_size`, where the same task (inference) is replicated to process multiple data items in parallel.
    *   **Message Passing**: SGLang avoids shared memory between processes and instead uses explicit message passing via ZMQ for coordination and control flow.
    *   **Collective Communications**: NCCL operations (like All-Reduce) are a prime example of collective communications, where a group of processes (GPUs) participate in a group-wide data operation.
    *   **Service Discovery**: The `nccl_port` and IPC/TCP addresses in `port_args.py` are a form of service discovery, allowing processes to find each other and connect.

    # Question 2:
    # What does `load_chat_template` mean? Why is it needed? What specific
    # template is being loaded? Provide an example.
    
    Answer:
    
    ### What it is
    The `load_chat_template` function configures the LLM with a specific format for handling conversational-style prompts.

    ### Why it is needed
    Most modern LLMs are fine-tuned for conversations. They are not just raw text completers; they are trained on text that is structured with special tokens and markers to distinguish between the "user," the "assistant," and "system" instructions.
    
    For example, a model might have been trained on data that looks like: `[INST] Who are you? [/INST] I am a helpful assistant. `
    
    If you send the raw prompt "Who are you?", the model might not respond correctly because it doesn't match the format it was trained on. A chat template **automatically formats your conversational input** into the exact string representation the model expects, ensuring high-quality responses.

    ### What it loads
    It loads a **Jinja2 template**. Jinja is a templating engine for Python. The template file (e.g., a `.jinja` file) contains the logic to assemble a conversation history into a single string, complete with the model's required special tokens (like `[INST]`, `<s>`, `<|end_of_text|>`, etc.). SGLang can use predefined templates (e.g., "llama-3") or a custom template file you provide.

    ### Example
    
    Let's say you want to have a conversation with Llama-3-Instruct and you provide this Python list of messages:
    
    ```python
    messages = [
        {"role": "system", "content": "You are a pirate."},
        {"role": "user", "content": "Who are you?"}
    ]
    ```

    The `chat_template` for Llama 3 looks something like this (simplified):
    
    ```jinja
    {% for message in messages %}{% if message['role'] == 'system' %}{{ '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% elif message['role'] == 'user' %}{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% elif message['role'] == 'assistant' %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}
    ```

    When `load_chat_template` is active, it uses this template to transform your simple message list into the following exact string, which is then sent to the model:

    ```
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are a pirate.<|eot_id|><|start_header_id|>user<|end_header_id|>

    Who are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    
    ```
    The model now sees the correctly formatted prompt and will generate its answer (e.g., "Arrr, I be a pirate of the high seas!") starting from the final `assistant` block.
    """
    # Configure global environment
    configure_logger(server_args)
    server_args.check_server_args()
    _set_envs_and_config(server_args)

    # Allocate ports for inter-process communications
    if port_args is None:
        port_args = PortArgs.init_new(server_args)
        logger.info(f"{server_args=}")

    # If using model from www.modelscope.cn, first download the model.
    server_args.model_path, server_args.tokenizer_path = prepare_model_and_tokenizer(
        server_args.model_path, server_args.tokenizer_path
    )

    scheduler_procs = []
    if server_args.dp_size == 1:
        # Launch tensor parallel scheduler processes
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=server_args.enable_memory_saver
        )

        scheduler_pipe_readers = []
        tp_size_per_node = server_args.tp_size // server_args.nnodes
        tp_rank_range = range(
            tp_size_per_node * server_args.node_rank,
            tp_size_per_node * (server_args.node_rank + 1),
        )
        for tp_rank in tp_rank_range:
            reader, writer = mp.Pipe(duplex=False)
            gpu_id = (
                server_args.base_gpu_id
                + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
            )
            proc = mp.Process(
                target=run_scheduler_process,
                args=(server_args, port_args, gpu_id, tp_rank, None, writer),
            )
            with memory_saver_adapter.configure_subprocess():
                proc.start()
            scheduler_procs.append(proc)
            scheduler_pipe_readers.append(reader)
    else:
        # Launch the data parallel controller
        reader, writer = mp.Pipe(duplex=False)
        scheduler_pipe_readers = [reader]
        proc = mp.Process(
            target=run_data_parallel_controller_process,
            args=(server_args, port_args, writer),
        )
        proc.start()
        scheduler_procs.append(proc)

    if server_args.node_rank >= 1:
        # In multi-node cases, non-zero rank nodes do not need to run tokenizer or detokenizer,
        # so they can just wait here.

        for reader in scheduler_pipe_readers:
            data = reader.recv()
            assert data["status"] == "ready"

        if os.getenv("SGLANG_BLOCK_NONZERO_RANK_CHILDREN") == "0":
            # When using `Engine` as a Python API, we don't want to block here.
            return None, None

        launch_dummy_health_check_server(server_args.host, server_args.port)

        for proc in scheduler_procs:
            proc.join()
            logger.error(
                f"Scheduler or DataParallelController {proc.pid} terminated with {proc.exitcode}"
            )
        return None, None

    # Launch detokenizer process
    detoken_proc = mp.Process(
        target=run_detokenizer_process,
        args=(
            server_args,
            port_args,
        ),
    )
    detoken_proc.start()

    # Launch tokenizer process
    tokenizer_manager = TokenizerManager(server_args, port_args)
    if server_args.chat_template:
        load_chat_template_for_openai_api(
            tokenizer_manager, server_args.chat_template, server_args.model_path
        )

    if server_args.completion_template:
        load_completion_template_for_openai_api(server_args.completion_template)

    # Wait for the model to finish loading
    scheduler_infos = []
    for i in range(len(scheduler_pipe_readers)):
        try:
            data = scheduler_pipe_readers[i].recv()
        except EOFError:
            logger.error(
                f"Rank {i} scheduler is dead. Please check if there are relevant logs."
            )
            scheduler_procs[i].join()
            logger.error(f"Exit code: {scheduler_procs[i].exitcode}")
            raise

        if data["status"] != "ready":
            raise RuntimeError(
                "Initialization failed. Please see the error messages above."
            )
        scheduler_infos.append(data)

    # Assume all schedulers have the same scheduler_info
    scheduler_info = scheduler_infos[0]
    tokenizer_manager.max_req_input_len = scheduler_info["max_req_input_len"]
    return tokenizer_manager, scheduler_info