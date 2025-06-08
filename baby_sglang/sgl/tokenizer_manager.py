class TokenizerManager:
    """TokenizerManager is a process that tokenizes the text."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        # Parse args
        self.server_args = server_args
        self.enable_metrics = server_args.enable_metrics
        self.log_requests = server_args.log_requests
        self.log_requests_level = server_args.log_requests_level

        # Init inter-process communication
        context = zmq.asyncio.Context(2)
        self.recv_from_detokenizer = get_zmq_socket(
            context, zmq.PULL, port_args.tokenizer_ipc_name, True
        )
        self.send_to_scheduler = get_zmq_socket(
            context, zmq.PUSH, port_args.scheduler_input_ipc_name, True
        )

        # Read model args
        self.model_path = server_args.model_path
        self.served_model_name = server_args.served_model_name
        self.model_config = ModelConfig(
            server_args.model_path,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
            context_length=server_args.context_length,
            model_override_args=server_args.json_model_override_args,
            is_embedding=server_args.is_embedding,
            enable_multimodal=server_args.enable_multimodal,
            dtype=server_args.dtype,
            quantization=server_args.quantization,
        )

        self.is_generation = self.model_config.is_generation
        self.is_image_gen = self.model_config.is_image_gen
        self.context_len = self.model_config.context_len
        self.image_token_id = self.model_config.image_token_id

        if self.model_config.is_multimodal:
            import_processors()
            _processor = get_processor(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
                use_fast=not server_args.disable_fast_image_processor,
            )

            # We want to parallelize the image pre-processing so we create an executor for it
            # We create mm_processor for any skip_tokenizer_init to make sure we still encode
            # images even with skip_tokenizer_init=False.
            self.mm_processor = get_mm_processor(
                self.model_config.hf_config, server_args, _processor
            )

            if server_args.skip_tokenizer_init:
                self.tokenizer = self.processor = None
            else:
                self.processor = _processor
                self.tokenizer = self.processor.tokenizer
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
        else:
            self.mm_processor = get_dummy_processor()

            if server_args.skip_tokenizer_init:
                self.tokenizer = self.processor = None
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )

        # Store states
        self.no_create_loop = False
        self.rid_to_state: Dict[str, ReqState] = {}
        self.gracefully_exit = False
        self.last_receive_tstamp = 0
        self.dump_requests_folder = ""  # By default do not dump
        self.dump_requests_threshold = 1000
        self.dump_request_list: List[Tuple] = []
        self.log_request_metadata = self.get_log_request_metadata()

        # The event to notify the weight sync is finished.
        self.model_update_lock = RWLock()
        self.model_update_result: Optional[Awaitable[UpdateWeightFromDiskReqOutput]] = (
            None
        )
        self.asyncio_tasks = set()

        # For session info
        self.session_futures = {}  # session_id -> asyncio event

        # Set after scheduler is initialized
        self.max_req_input_len = None

        # Metrics
        if self.enable_metrics:
            self.metrics_collector = TokenizerMetricsCollector(
                labels={
                    "model_name": self.server_args.served_model_name,
                    # TODO: Add lora name/path in the future,
                },
            )

        # Communicators
        self.init_weights_update_group_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.update_weights_from_distributed_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.update_weights_from_tensor_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.get_weights_by_name_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.release_memory_occupation_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.resume_memory_occupation_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.flush_cache_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.start_profile_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.get_internal_state_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.expert_distribution_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )

        self._result_dispatcher = TypeBasedDispatcher(
            [
                (
                    (
                        BatchStrOut,
                        BatchEmbeddingOut,
                        BatchTokenIDOut,
                        BatchMultimodalOut,
                    ),
                    self._handle_batch_output,
                ),
                (OpenSessionReqOutput, self._handle_open_session_req_output),
                (
                    UpdateWeightFromDiskReqOutput,
                    self._handle_update_weights_from_disk_req_output,
                ),
                (
                    InitWeightsUpdateGroupReqOutput,
                    self.init_weights_update_group_communicator.handle_recv,
                ),
                (
                    UpdateWeightsFromDistributedReqOutput,
                    self.update_weights_from_distributed_communicator.handle_recv,
                ),
                (
                    UpdateWeightsFromTensorReqOutput,
                    self.update_weights_from_tensor_communicator.handle_recv,
                ),
                (
                    GetWeightsByNameReqOutput,
                    self.get_weights_by_name_communicator.handle_recv,
                ),
                (
                    ReleaseMemoryOccupationReqOutput,
                    self.release_memory_occupation_communicator.handle_recv,
                ),
                (
                    ResumeMemoryOccupationReqOutput,
                    self.resume_memory_occupation_communicator.handle_recv,
                ),
                (
                    FlushCacheReqOutput,
                    self.flush_cache_communicator.handle_recv,
                ),
                (
                    ProfileReqOutput,
                    self.start_profile_communicator.handle_recv,
                ),
                (
                    GetInternalStateReqOutput,
                    self.get_internal_state_communicator.handle_recv,
                ),
                (
                    ExpertDistributionReqOutput,
                    self.expert_distribution_communicator.handle_recv,
                ),
                (HealthCheckOutput, lambda x: None),
            ]
        )

        self.disaggregation_mode = DisaggregationMode(
            self.server_args.disaggregation_mode
        )
        self.transfer_backend = TransferBackend(
            self.server_args.disaggregation_transfer_backend
        )
        # for disaggregtion, start kv boostrap server on prefill
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # only start bootstrap server on prefill tm
            kv_bootstrap_server_class = get_kv_class(
                self.transfer_backend, KVClassType.BOOTSTRAP_SERVER
            )
            self.bootstrap_server = kv_bootstrap_server_class(
                self.server_args.disaggregation_bootstrap_port
            )

    async def generate_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
    ):
        created_time = time.time()

        self.auto_create_handle_loop()

        if isinstance(obj, EmbeddingReqInput) and self.is_generation:
            raise ValueError(
                "This model does not appear to be an embedding model by default. "
                "Please add `--is-embedding` when launching the server or try another model."
            )

        obj.normalize_batch_and_arguments()

        if self.log_requests:
            max_length, skip_names, _ = self.log_request_metadata
            logger.info(
                f"Receive: obj={dataclass_to_string_truncated(obj, max_length, skip_names=skip_names)}"
            )

        async with self.model_update_lock.reader_lock:
            is_single = obj.is_single
            if is_single:
                tokenized_obj = await self._tokenize_one_request(obj)
                self._send_one_request(obj, tokenized_obj, created_time)
                async for response in self._wait_one_response(obj, request):
                    yield response
            else:
                async for response in self._handle_batch_request(
                    obj, request, created_time
                ):
                    yield response