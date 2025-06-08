def load_chat_template_for_openai_api(tokenizer_manager, chat_template_arg, model_path):
    global chat_template_name

    logger.info(
        f"Use chat template for the OpenAI-compatible API server: {chat_template_arg}"
    )

    if not chat_template_exists(chat_template_arg):
        if not os.path.exists(chat_template_arg):
            raise RuntimeError(
                f"Chat template {chat_template_arg} is not a built-in template name "
                "or a valid chat template file path."
            )
        if chat_template_arg.endswith(".jinja"):
            with open(chat_template_arg, "r") as f:
                chat_template = "".join(f.readlines()).strip("\n")
            tokenizer_manager.tokenizer.chat_template = chat_template.replace(
                "\\n", "\n"
            )
            chat_template_name = None
        else:
            assert chat_template_arg.endswith(
                ".json"
            ), "unrecognized format of chat template file"
            with open(chat_template_arg, "r") as filep:
                template = json.load(filep)
                try:
                    sep_style = SeparatorStyle[template["sep_style"]]
                except KeyError:
                    raise ValueError(
                        f"Unknown separator style: {template['sep_style']}"
                    ) from None
                register_conv_template(
                    Conversation(
                        name=template["name"],
                        system_template=template["system"] + "\n{system_message}",
                        system_message=template.get("system_message", ""),
                        roles=(template["user"], template["assistant"]),
                        sep_style=sep_style,
                        sep=template.get("sep", "\n"),
                        stop_str=template["stop_str"],
                    ),
                    override=True,
                )
            chat_template_name = template["name"]
    else:
        chat_template_name = chat_template_arg