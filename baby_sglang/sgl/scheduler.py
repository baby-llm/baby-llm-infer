def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
):
    # Generate the prefix
    if dp_rank is None:
        prefix = f" TP{tp_rank}"
    else:
        prefix = f" DP{dp_rank} TP{tp_rank}"

    # Config the process
    kill_itself_when_parent_died()
    setproctitle.setproctitle(f"sglang::scheduler{prefix.replace(' ', '_')}")
    faulthandler.enable()
    parent_process = psutil.Process().parent()

    # [For Router] if env var "SGLANG_DP_RANK" exist, set dp_rank to the value of the env var
    if dp_rank is None and "SGLANG_DP_RANK" in os.environ:
        dp_rank = int(os.environ["SGLANG_DP_RANK"])

    # Configure the logger
    configure_logger(server_args, prefix=prefix)
    suppress_other_loggers()

    # Set cpu affinity to this gpu process
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(server_args.tp_size, server_args.nnodes, gpu_id)

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
        disaggregation_mode: DisaggregationMode = scheduler.disaggregation_mode

        if disaggregation_mode == DisaggregationMode.NULL:
            if scheduler.enable_overlap:
                scheduler.event_loop_overlap()
            else:
                scheduler.event_loop_normal()
        elif disaggregation_mode == DisaggregationMode.PREFILL:
            if scheduler.enable_overlap:
                scheduler.event_loop_overlap_disagg_prefill()
            else:
                scheduler.event_loop_normal_disagg_prefill()
        elif disaggregation_mode == DisaggregationMode.DECODE:
            if scheduler.enable_overlap:
                scheduler.event_loop_overlap_disagg_decode()
            else:
                scheduler.event_loop_normal_disagg_decode()

    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)