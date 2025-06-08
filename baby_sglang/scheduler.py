import psutil
import setproctitle
import torch
import zmq
import setproctitle
import faulthandler
import logging
import signal
from typing import Optional
from baby_sglang.snippet.server_args import ServerArgs, PortArgs
from baby_sglang.utils import (
    get_zmq_socket,
)
from utils import get_exception_traceback
from model_config import ModelConfig
from hf_transformers_utils import get_tokenizer
from tp_worker import TpModelWorker

logger = logging.getLogger(__name__)

def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
):
    setproctitle.setproctitle("sglang::scheduler")
    faulthandler.enable()

    # [For Router] if env var "SGLANG_DP_RANK" exist, set dp_rank to the value of the env var
    # if dp_rank is None and "SGLANG_DP_RANK" in os.environ:
    #     dp_rank = int(os.environ["SGLANG_DP_RANK"])

    # Configue the logger
    # if dp_rank is None:
    #     configure_logger(server_args, prefix=f" TP{tp_rank}")
    # else:
    #     configure_logger(server_args, prefix=f" DP{dp_rank} TP{tp_rank}")
    # suppress_other_loggers()

    # Set cpu affinity to this gpu process
    # if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        # set_gpu_proc_affinity(server_args.tp_size, server_args.nnodes, gpu_id)

    parent_process = psutil.Process().parent()

    # Create a scheduler and run the event loop
    try:
        scheduler = Scheduler(server_args, port_args, gpu_id, tp_rank, dp_rank)
        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": scheduler.max_total_num_tokens,
                "max_req_input_len": scheduler.max_req_input_len,
            }
        )
        # if scheduler.enable_overlap:
            # scheduler.event_loop_overlap()
        # else:
        scheduler.event_loop_normal()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
  
class Scheduler:
    """A scheduler that manages a tensor parallel GPU worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
    ):
        # Parse args
        self.server_args = server_args
        self.tp_rank = tp_rank
        self.tp_size = server_args.tp_size
        self.schedule_policy = server_args.schedule_policy
        # self.disable_jump_forward = server_args.disable_jump_forward
        # self.lora_paths = server_args.lora_paths
        # self.max_loras_per_batch = server_args.max_loras_per_batch
        # self.enable_overlap = not server_args.disable_overlap_schedule
        # self.skip_tokenizer_init = server_args.skip_tokenizer_init
        # self.enable_metrics = server_args.enable_metrics
        # self.spec_algorithm = SpeculativeAlgorithm.from_string(
            # server_args.speculative_algorithm
        # )
        self.decode_mem_cache_buf_multiplier = 1
        # (
        #     self.server_args.speculative_num_draft_tokens
        #     if not self.spec_algorithm.is_none()
        #     else 1
        # )

        # Distributed rank info
        self.dp_size = server_args.dp_size
        self.attn_tp_rank, self.attn_tp_size, self.dp_rank = self.tp_rank, self.tp_size, 0
        # (
        #     compute_dp_attention_world_info(
        #         # server_args.enable_dp_attention,
        #         False,
        #         self.tp_rank,
        #         self.tp_size,
        #         self.dp_size,
        #     )
        # )

        # Init inter-process communication
        context = zmq.Context(2)
        # if self.attn_tp_rank == 0:
        self.recv_from_tokenizer = get_zmq_socket(
            context, zmq.PULL, port_args.scheduler_input_ipc_name, False
        )
        self.send_to_tokenizer = get_zmq_socket(
            context, zmq.PUSH, port_args.tokenizer_ipc_name, False
        )

        # if server_args.skip_tokenizer_init:
        #     # Directly send to the TokenizerManager
        #     self.send_to_detokenizer = get_zmq_socket(
        #         context, zmq.PUSH, port_args.tokenizer_ipc_name, False
        #     )
        # else:
        # Send to the DetokenizerManager
        self.send_to_detokenizer = get_zmq_socket(
            context, zmq.PUSH, port_args.detokenizer_ipc_name, False
        )
        # else:
        #     self.recv_from_tokenizer = None
        #     self.send_to_tokenizer = SimpleNamespace(send_pyobj=lambda x: None)
        #     self.send_to_detokenizer = SimpleNamespace(send_pyobj=lambda x: None)

        # Init tokenizer
        # self.model_config = ModelConfig(
        #     server_args.model_path,
        #     trust_remote_code=server_args.trust_remote_code,
        #     revision=server_args.revision,
        #     context_length=server_args.context_length,
        #     model_override_args=server_args.json_model_override_args,
        #     is_embedding=server_args.is_embedding,
        #     dtype=server_args.dtype,
        #     quantization=server_args.quantization,
        # )
        # self.is_generation = self.model_config.is_generation

        # if server_args.skip_tokenizer_init:
        #     self.tokenizer = self.processor = None
        # else:
        #     if self.model_config.is_multimodal:
        #         self.processor = get_processor(
        #             server_args.tokenizer_path,
        #             tokenizer_mode=server_args.tokenizer_mode,
        #             trust_remote_code=server_args.trust_remote_code,
        #             revision=server_args.revision,
        #         )
        #         self.tokenizer = self.processor.tokenizer
        #     else:
        #         self.tokenizer = get_tokenizer(
        #             server_args.tokenizer_path,
        #             tokenizer_mode=server_args.tokenizer_mode,
        #             trust_remote_code=server_args.trust_remote_code,
        #             revision=server_args.revision,
        #         )
        self.model_config = ModelConfig(
            server_args.model_path,
            trust_remote_code=server_args.trust_remote_code,
            # revision=server_args.revision,
            # context_length=server_args.context_length,
            # model_override_args=server_args.json_model_override_args,
            # is_embedding=server_args.is_embedding,
            # enable_multimodal=server_args.enable_multimodal,
            # dtype=server_args.dtype,
            # quantization=server_args.quantization,
        )

        self.is_generation = self.model_config.is_generation
        
        self.tokenizer = get_tokenizer(
            server_args.tokenizer_path,
            tokenizer_mode=server_args.tokenizer_mode,
            trust_remote_code=server_args.trust_remote_code,
            # revision=server_args.revision,
        )

        # Check whether overlap can be enabled
        # if not self.is_generation:
        #     self.enable_overlap = False
        #     logger.info("Overlap scheduler is disabled for embedding models.")

        # if self.model_config.is_multimodal:
        #     self.enable_overlap = False
        #     logger.info("Overlap scheduler is disabled for multimodal models.")

        # if self.enable_overlap:
        #     self.disable_jump_forward = True

        # Launch a tensor parallel worker
        # if self.enable_overlap:
        #     TpWorkerClass = TpModelWorkerClient
        # else:
        TpWorkerClass = TpModelWorker

        self.tp_worker = TpWorkerClass(
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            nccl_port=port_args.nccl_port,
        )

        # Launch a worker for speculative decoding if needed
        # if self.spec_algorithm.is_eagle():
        #     from sglang.srt.speculative.eagle_worker import EAGLEWorker

        #     self.draft_worker = EAGLEWorker(
        #         gpu_id=gpu_id,
        #         tp_rank=tp_rank,
        #         server_args=server_args,
        #         nccl_port=port_args.nccl_port,
        #         target_worker=self.tp_worker,
        #         dp_rank=dp_rank,
        #     )
        # else:
        self.draft_worker = None

        # Get token and memory info from the model worker
        (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_req_len,
            self.max_req_input_len,
            self.random_seed,
            self.device,
            worker_global_server_args_dict,
            _,
            _,
            _,
        ) = self.tp_worker.get_worker_info()
        self.tp_cpu_group = self.tp_worker.get_tp_cpu_group()
        self.attn_tp_cpu_group = self.tp_worker.get_attention_tp_cpu_group()
        self.pad_input_ids_func = self.tp_worker.get_pad_input_ids_func()
        global_server_args_dict.update(worker_global_server_args_dict)
        set_random_seed(self.random_seed)
        # Print debug info
        logger.info(
            f"max_total_num_tokens={self.max_total_num_tokens}, "
            f"max_prefill_tokens={self.max_prefill_tokens}, "
            f"max_running_requests={self.max_running_requests}, "
            f"context_len={self.model_config.context_len}"
        )

        # Init memory pool and cache
        self.req_to_token_pool, self.token_to_kv_pool = self.tp_worker.get_memory_pool()

        if (
            server_args.chunked_prefill_size is not None
            and server_args.disable_radix_cache
        ):
            self.tree_cache = ChunkCache(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool=self.token_to_kv_pool,
            )
        else:
            self.tree_cache = RadixCache(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool=self.token_to_kv_pool,
                disable=server_args.disable_radix_cache,
            )
        self.tree_cache_metrics = {"total": 0, "hit": 0}
        self.policy = SchedulePolicy(self.schedule_policy, self.tree_cache)

        # Init running status
        self.waiting_queue: List[Req] = []
        # The running decoding batch for continuous batching
        self.running_batch: Optional[ScheduleBatch] = None
        # The current forward batch
        self.cur_batch: Optional[ScheduleBatch] = None
        # The current forward batch
        self.last_batch: Optional[ScheduleBatch] = None
        self.forward_ct = 0
        self.forward_ct_decode = 0
        self.num_generated_tokens = 0
        self.spec_num_total_accepted_tokens = 0
        self.spec_num_total_forward_ct = 0
        self.last_decode_stats_tic = time.time()
        self.stream_interval = server_args.stream_interval
        self.current_stream = torch.get_device_module(self.device).current_stream()
        if self.device == "cpu":
            self.current_stream.synchronize = lambda: None  # No-op for CPU

        # Session info
        # self.sessions: Dict[str, Session] = {}

        # Init chunked prefill
        # self.chunked_prefill_size = server_args.chunked_prefill_size
        # if self.chunked_prefill_size <= 0:  # -1 means disable
        self.chunked_prefill_size = None
        self.being_chunked_req = None
        self.is_mixed_chunk = False
        # (
        #     self.chunked_prefill_size is not None and server_args.enable_mixed_chunk 
        # )

        # Init the grammar backend for constrained generation
        self.grammar_queue: List[Req] = []
        # if not server_args.skip_tokenizer_init:
        #     self.grammar_backend = create_grammar_backend(
        #         server_args, self.tokenizer, self.model_config.vocab_size
        #     )
        # else:
        self.grammar_backend = None

        # Init new token estimation
        assert (
            server_args.schedule_conservativeness >= 0
        ), "Invalid schedule_conservativeness"

        self.init_new_token_ratio = min(
            global_config.default_init_new_token_ratio
            * server_args.schedule_conservativeness,
            1.0,
        )
        self.min_new_token_ratio = min(
            self.init_new_token_ratio
            * global_config.default_min_new_token_ratio_factor,
            1.0,
        )
        self.new_token_ratio_decay = (
            self.init_new_token_ratio - self.min_new_token_ratio
        ) / global_config.default_new_token_ratio_decay_steps
        self.new_token_ratio = self.init_new_token_ratio

        # Tells whether the current running batch is full so that we can skip
        # the check of whether to prefill new requests.
        # This is an optimization to reduce the overhead of the prefill check.
        self.batch_is_full = False

        # Init watchdog thread
        self.watchdog_timeout = server_args.watchdog_timeout
        t = threading.Thread(target=self.watchdog_thread, daemon=True)
        t.start()
        self.parent_process = psutil.Process().parent()

        # self.memory_saver_adapter = TorchMemorySaverAdapter.create(
        #     enable=server_args.enable_memory_saver
        # )

        # Init profiler
        if os.getenv("SGLANG_TORCH_PROFILER_DIR", "") == "":
            self.profiler = None
        else:
            self.torch_profiler_trace_dir = os.getenv("SGLANG_TORCH_PROFILER_DIR")
            logger.info(
                "Profiling enabled. Traces will be saved to: %s",
                self.torch_profiler_trace_dir,
            )
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_stack=True,
            )

        # Init metrics stats
        # self.stats = SchedulerStats()
        # if self.enable_metrics:
        #     self.metrics_collector = SchedulerMetricsCollector(
        #         labels={
        #             "model_name": self.server_args.served_model_name,
        #             # TODO: Add lora name/path in the future,
        #         },
        #     )

        # Init request dispatcher
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateReqInput, self.handle_generate_request),
                # (TokenizedEmbeddingReqInput, self.handle_embedding_request),
                # (FlushCacheReq, self.flush_cache_wrapped),
                # (AbortReq, self.abort_request),
                # (UpdateWeightFromDiskReqInput, self.update_weights_from_disk),
                # (InitWeightsUpdateGroupReqInput, self.init_weights_update_group),
                # (
                #     UpdateWeightsFromDistributedReqInput,
                #     self.update_weights_from_distributed,
                # ),
                # (UpdateWeightsFromTensorReqInput, self.update_weights_from_tensor),
                # (GetWeightsByNameReqInput, self.get_weights_by_name),
                # (ProfileReq, self.profile),
                # (OpenSessionReqInput, self.open_session),
                # (CloseSessionReqInput, self.close_session),
                # (
                #     ReleaseMemoryOccupationReqInput,
                #     lambda _: self.release_memory_occupation(),
                # ),
                # (
                #     ResumeMemoryOccupationReqInput,
                #     lambda _: self.resume_memory_occupation(),
                # ),
            ]
        )