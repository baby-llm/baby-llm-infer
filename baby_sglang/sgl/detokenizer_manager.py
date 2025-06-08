def run_detokenizer_process(
    server_args: ServerArgs,
    port_args: PortArgs,
):
    kill_itself_when_parent_died()
    setproctitle.setproctitle("sglang::detokenizer")
    configure_logger(server_args)
    parent_process = psutil.Process().parent()

    try:
        manager = DetokenizerManager(server_args, port_args)
        manager.event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"DetokenizerManager hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)