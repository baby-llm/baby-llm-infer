from baby_sglang.model_config import ModelConfig
from baby_sglang.snippet.server_args import ServerArgs

class ModelRunner:
    """ModelRunner runs the forward passes of the models."""

    def __init__(
        self,
        model_config: ModelConfig,
        # mem_fraction_static: float,
        gpu_id: int,
        tp_rank: int,
        tp_size: int,
        nccl_port: int,
        server_args: ServerArgs,
        is_draft_worker: bool = False,
    ):
        # Parse args
        self.model_config = model_config
        # self.mem_fraction_static = mem_fraction_static
        self.device = server_args.device
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.dist_port = nccl_port
        self.server_args = server_args
        self.is_draft_worker = is_draft_worker
        self.is_generation = model_config.is_generation
        # self.is_multimodal = model_config.is_multimodal
        self.should_log = tp_rank == 0
        # self.spec_algorithm = SpeculativeAlgorithm.from_string(
        #     server_args.speculative_algorithm
        # )

        # Model-specific adjustment
        # if (
        #     self.model_config.attention_arch == AttentionArch.MLA
        #     and not self.server_args.disable_mla
        # ):
        #     # TODO: add MLA optimization on CPU
        #     if self.server_args.device != "cpu":
        #         logger.info("MLA optimization is turned on. Use triton backend.")
        #         self.server_args.attention_backend = "triton"

        # if self.server_args.enable_double_sparsity:
        #     logger.info(
        #         "Double sparsity optimization is turned on. Use triton backend without CUDA graph."
        #     )
        #     self.server_args.attention_backend = "triton"
        #     self.server_args.disable_cuda_graph = True
        #     if self.server_args.ds_heavy_channel_type is None:
        #         raise ValueError(
        #             "Please specify the heavy channel type for double sparsity optimization."
        #         )
        #     self.init_double_sparsity_channel_config(
        #         self.server_args.ds_heavy_channel_type
        #     )

        # if self.is_multimodal:
        #     self.mem_fraction_static *= 0.95
        #     logger.info(
        #         f"Automatically reduce --mem-fraction-static to {self.mem_fraction_static:.3f} "
        #         f"because this is a multimodal model."
        #     )

        #     if self.model_config.hf_config.architectures == [
        #         "MllamaForConditionalGeneration"
        #     ]:
        #         logger.info("Automatically turn off --chunked-prefill-size for mllama.")
        #         server_args.chunked_prefill_size = -1

        #     if self.model_config.hf_config.architectures == [
        #         "Qwen2VLForConditionalGeneration"
        #     ]:
        #         # TODO: qwen2-vl does not support radix cache now, set disable_radix_cache=True automatically
        #         logger.info(
        #             "Automatically turn off --chunked-prefill-size and disable radix cache for qwen2-vl."
        #         )
        #         server_args.chunked_prefill_size = -1
        #         server_args.disable_radix_cache = True

        # Global vars
        # if server_args.show_time_cost:
        #     enable_show_time_cost()
        # if server_args.disable_outlines_disk_cache:
        #     from outlines.caching import disable_cache

        #     disable_cache()

        global_server_args_dict.update(
            {
                "attention_backend": server_args.attention_backend,
                "sampling_backend": server_args.sampling_backend,
                "triton_attention_reduce_in_fp32": server_args.triton_attention_reduce_in_fp32,
                # "disable_mla": server_args.disable_mla,
                # "torchao_config": server_args.torchao_config,
                "enable_nan_detection": server_args.enable_nan_detection,
                "enable_dp_attention": server_args.enable_dp_attention,
                "enable_ep_moe": server_args.enable_ep_moe,
                "device": server_args.device,
            }
        )

        set_cpu_offload_max_bytes(int(server_args.cpu_offload_gb * 1024**3))

        # Get memory before model loading
        min_per_gpu_memory = self.init_torch_distributed()

        # self.memory_saver_adapter = TorchMemorySaverAdapter.create(
        #     enable=self.server_args.enable_memory_saver
        # )

        # Load the model
        self.sampler = Sampler()
        self.load_model()

        # Apply torchao quantization
        # apply_torchao_config_to_model(
        #     self.model, global_server_args_dict["torchao_config"]
        # )

        # Apply torch TP if the model supports it
        # supports_torch_tp = getattr(self.model, "supports_torch_tp", False)
        # if self.tp_size > 1 and supports_torch_tp:
        #     self.apply_torch_tp()
        #     self.torch_tp_applied = True
        # else:
        #     self.torch_tp_applied = False

        # Init memory pool and attention backends
        # if server_args.lora_paths is not None:
        #     self.init_lora_manager()
        self.init_memory_pool(
            min_per_gpu_memory,
            server_args.max_running_requests,
            server_args.max_total_tokens,
        )
        if self.device == "cuda":
            self.init_cublas()
            self.init_attention_backend()
            self.init_cuda_graphs()
        else:
            self.cuda_graph_runner = None
            self.init_attention_backend()
    
    