from typing import Dict, Optional, Type, Union
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

CONTEXT_LENGTH_KEYS = [
    "max_sequence_length",
    "seq_length",
    "max_seq_len",
    "model_max_length",
    "max_position_embeddings",
]

def get_config(
    model: str,
    trust_remote_code: bool,
    # revision: Optional[str] = None,
    # model_override_args: Optional[dict] = None,
    **kwargs,
):
    # IGNORE: gguf
    # is_gguf = check_gguf_file(model)
    # if is_gguf:
    #     kwargs["gguf_file"] = model
    #     model = Path(model).parent

    config = AutoConfig.from_pretrained(
        model, trust_remote_code=trust_remote_code, 
        # revision=revision, 
        **kwargs
    )
    # if config.model_type in _CONFIG_REGISTRY:
    #     config_class = _CONFIG_REGISTRY[config.model_type]
    #     config = config_class.from_pretrained(model, revision=revision)
    #     # NOTE(HandH1998): Qwen2VL requires `_name_or_path` attribute in `config`.
    #     setattr(config, "_name_or_path", model)
    # if model_override_args:
        # config.update(model_override_args)

    # Special architecture mapping check for GGUF models
    # if is_gguf:
    #     if config.model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
    #         raise RuntimeError(f"Can't get gguf config for {config.model_type}.")
    #     model_type = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type]
    #     config.update({"architectures": [model_type]})

    return config

def get_context_length(config):
    """Get the context length of a model from a huggingface model configs."""
    text_config = config
    rope_scaling = getattr(text_config, "rope_scaling", None)
    if rope_scaling:
        rope_scaling_factor = rope_scaling.get("factor", 1)
        if "original_max_position_embeddings" in rope_scaling:
            rope_scaling_factor = 1
        if rope_scaling.get("rope_type", None) == "llama3":
            rope_scaling_factor = 1
    else:
        rope_scaling_factor = 1

    for key in CONTEXT_LENGTH_KEYS:
        val = getattr(text_config, key, None)
        if val is not None:
            return int(rope_scaling_factor * val)
    return 2048

def get_tokenizer(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    # tokenizer_revision: Optional[str] = None,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Gets a tokenizer for the given model name via Huggingface."""
    # if tokenizer_mode == "slow":
    #     if kwargs.get("use_fast", False):
    #         raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")
    #     kwargs["use_fast"] = False

    # is_gguf = check_gguf_file(tokenizer_name)
    # if is_gguf:
    #     kwargs["gguf_file"] = tokenizer_name
    #     tokenizer_name = Path(tokenizer_name).parent

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            *args,
            trust_remote_code=trust_remote_code,
            # tokenizer_revision=tokenizer_revision,
            clean_up_tokenization_spaces=False,
            **kwargs,
        )
    except TypeError as e:
        # The LLaMA tokenizer causes a protobuf error in some environments.
        err_msg = (
            "Failed to load the tokenizer."
        )
        raise RuntimeError(err_msg) from e
    except ValueError as e:
        # If the error pertains to the tokenizer class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        if not trust_remote_code and (
            "does not exist or is not currently imported." in str(e)
            or "requires you to execute the tokenizer file" in str(e)
        ):
            err_msg = (
                "Failed to load the tokenizer. If the tokenizer is a custom "
                "tokenizer not yet available in the HuggingFace transformers "
                "library, consider setting `trust_remote_code=True` in LLM "
                "or using the `--trust-remote-code` flag in the CLI."
            )
            raise RuntimeError(err_msg) from e
        else:
            raise e

    # if not isinstance(tokenizer, PreTrainedTokenizerFast):
    #     warnings.warn(
    #         "Using a slow tokenizer. This might cause a significant "
    #         "slowdown. Consider using a fast tokenizer instead."
    #     )

    attach_additional_stop_token_ids(tokenizer)
    return tokenizer

def attach_additional_stop_token_ids(tokenizer):
    # Special handling for stop token <|eom_id|> generated by llama 3 tool use.
    if "<|eom_id|>" in tokenizer.get_added_vocab():
        tokenizer.additional_stop_token_ids = set(
            [tokenizer.get_added_vocab()["<|eom_id|>"]]
        )
    else:
        tokenizer.additional_stop_token_ids = None