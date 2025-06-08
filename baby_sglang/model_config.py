from typing import List, Optional, Set, Union
from enum import IntEnum, auto
from transformers import PretrainedConfig
import torch

from baby_sglang.hf_transformers_utils import get_config, get_context_length

class AttentionArch(IntEnum):
    MLA = auto()
    MHA = auto()

class ModelConfig:
    def __init__(
        self,
        model_path: str,
        trust_remote_code: bool = True,
        # revision: Optional[str] = None,
        # context_length: Optional[int] = None,
        # model_override_args: Optional[dict] = None,
        # is_embedding: Optional[bool] = None,
        # dtype: str = "auto",
        # quantization: Optional[str] = None,
    ) -> None:
        self.model_path = model_path
        # self.revision = revision
        # self.quantization = quantization

        # Parse args
        # self.model_override_args = json.loads(model_override_args)
        self.hf_config = get_config(
            model_path,
            trust_remote_code=trust_remote_code,
            # revision=revision,
            # model_override_args=self.model_override_args,
        )
        self.hf_text_config = self.hf_config
        # self.hf_text_config = get_hf_text_config(self.hf_config)

        # Check model type
        self.is_generation = True
        # self.is_generation = is_generation_model(
        #     self.hf_config.architectures, is_embedding
        # )
        # self.is_multimodal = is_multimodal_model(self.hf_config.architectures)
        # self.is_encoder_decoder = is_encoder_decoder_model(self.hf_config.architectures)
        self.dtype = _get_and_verify_dtype(self.hf_text_config)

        # Derive context length
        derived_context_len = get_context_length(self.hf_text_config)
        # if context_length is not None:
        #     if context_length > derived_context_len:
        #         if get_bool_env_var("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"):
        #             logger.warning(
        #                 f"Warning: User-specified context_length ({context_length}) is greater than the derived context_length ({derived_context_len}). "
        #                 f"This may lead to incorrect model outputs or CUDA errors."
        #             )
        #             self.context_len = context_length
        #         else:
        #             raise ValueError(
        #                 f"User-specified context_length ({context_length}) is greater than the derived context_length ({derived_context_len}). "
        #                 f"This may lead to incorrect model outputs or CUDA errors. Note that the derived context_length may differ from max_position_embeddings in the model's config. "
        #                 f"To allow overriding this maximum, set the env var SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"
        #             )
        #     else:
        #         self.context_len = context_length
        # else:
        self.context_len = derived_context_len

        # Unify the config keys for hf_text_config
        self.head_dim = getattr(
            self.hf_text_config,
            "head_dim",
            self.hf_text_config.hidden_size // self.hf_text_config.num_attention_heads,
        )

        # FIXME: temporary special judge for MLA architecture
        # if (
        #     "DeepseekV2ForCausalLM" in self.hf_config.architectures
        #     or "DeepseekV3ForCausalLM" in self.hf_config.architectures
        # ):
        #     self.head_dim = 256
        #     self.attention_arch = AttentionArch.MLA
        #     self.kv_lora_rank = self.hf_config.kv_lora_rank
        #     self.qk_rope_head_dim = self.hf_config.qk_rope_head_dim
        # elif "MiniCPM3ForCausalLM" in self.hf_config.architectures:
        #     self.head_dim = 128
        #     self.attention_arch = AttentionArch.MLA
        #     self.kv_lora_rank = self.hf_config.kv_lora_rank
        #     self.qk_rope_head_dim = self.hf_config.qk_rope_head_dim
        # else:
        # IGNORE: fixed MHA
        self.attention_arch = AttentionArch.MHA

        self.num_attention_heads = self.hf_text_config.num_attention_heads
        self.num_key_value_heads = getattr(
            self.hf_text_config, "num_key_value_heads", None
        )

        # for Dbrx and MPT models
        # if self.hf_config.model_type in ["dbrx", "mpt"]:
        #     self.num_key_value_heads = getattr(
        #         self.hf_config.attn_config, "kv_n_heads", None
        #     )

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        self.hidden_size = self.hf_text_config.hidden_size
        self.num_hidden_layers = self.hf_text_config.num_hidden_layers
        self.vocab_size = self.hf_text_config.vocab_size

        # Verify quantization
        # self._verify_quantization()

        # Cache attributes
        self.hf_eos_token_id = self.get_hf_eos_token_id()
        # self.image_token_id = getattr(self.hf_config, "image_token_id", None)
    
    def get_hf_eos_token_id(self) -> Optional[Set[int]]:
        eos_ids = getattr(self.hf_config, "eos_token_id", None)
        if eos_ids:
            # it can be either int or list of int
            eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
        return eos_ids

def _get_and_verify_dtype(
    config: PretrainedConfig,
    # dtype: Union[str, torch.dtype],
) -> torch.dtype:
    # NOTE: getattr(config, "torch_dtype", torch.float32) is not correct
    # because config.torch_dtype can be None.
    config_dtype = getattr(config, "torch_dtype", None)
    if config_dtype is None:
        config_dtype = torch.float32

    # IGNORE: self-defined dtype
    # if isinstance(dtype, str):
    #     dtype = dtype.lower()
    #     if dtype == "auto":
    #         if config_dtype == torch.float32:
    #             if config.model_type == "gemma2":
    #                 logger.info(
    #                     "For Gemma 2, we downcast float32 to bfloat16 instead "
    #                     "of float16 by default. Please specify `dtype` if you "
    #                     "want to use float16."
    #                 )
    #                 torch_dtype = torch.bfloat16
    #             else:
    #                 # Following the common practice, we use float16 for float32
    #                 # models.
    #                 torch_dtype = torch.float16
    #         else:
    #             torch_dtype = config_dtype
    #     else:
    #         if dtype not in _STR_DTYPE_TO_TORCH_DTYPE:
    #             raise ValueError(f"Unknown dtype: {dtype}")
    #         torch_dtype = _STR_DTYPE_TO_TORCH_DTYPE[dtype]
    # elif isinstance(dtype, torch.dtype):
    #     torch_dtype = dtype
    # else:
    #     raise ValueError(f"Unknown dtype: {dtype}")

    # # Verify the dtype.
    # if torch_dtype != config_dtype:
        if torch_dtype == torch.float32:
            # Upcasting to float32 is allowed.
            logger.info("Upcasting %s to %s.", config_dtype, torch_dtype)
            pass
        elif config_dtype == torch.float32:
            # Downcasting from float32 to float16 or bfloat16 is allowed.
            logger.info("Downcasting %s to %s.", config_dtype, torch_dtype)
            pass
        else:
            # Casting between float16 and bfloat16 is allowed with a warning.
            logger.warning("Casting %s to %s.", config_dtype, torch_dtype)

    # return torch_dtype
    return config_dtype