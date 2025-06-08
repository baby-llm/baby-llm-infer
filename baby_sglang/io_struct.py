from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union
from schedule_batch import BaseFinishReason
from sampling_params import SamplingParams
import uuid

@dataclass
class BatchStrOut:
    # The request id
    rids: List[str]
    # The finish reason
    finished_reasons: List[dict]
    # The output decoded strings
    output_strs: List[str]

    # Token counts
    prompt_tokens: List[int]
    completion_tokens: List[int]
    cached_tokens: List[int]

    # Logprobs
    # input_token_logprobs_val: List[float]
    # input_token_logprobs_idx: List[int]
    # output_token_logprobs_val: List[float]
    # output_token_logprobs_idx: List[int]
    # input_top_logprobs_val: List[List]
    # input_top_logprobs_idx: List[List]
    # output_top_logprobs_val: List[List]
    # output_top_logprobs_idx: List[List]

@dataclass
class BatchTokenIDOut:
    # The request id
    rids: List[str]
    # The finish reason
    finished_reasons: List[BaseFinishReason]
    # For incremental decoding
    # The version id to sync decode status with in detokenizer_manager
    vids: List[int]
    decoded_texts: List[str]
    decode_ids: List[int]
    read_offsets: List[int]
    # Only used when `--skip-tokenizer-init` is on
    output_ids: Optional[List[int]]
    # Detokenization configs
    skip_special_tokens: List[bool]
    spaces_between_special_tokens: List[bool]
    no_stop_trim: List[bool]
    # Token counts
    prompt_tokens: List[int]
    completion_tokens: List[int]
    cached_tokens: List[int]
    # Logprobs
    # input_token_logprobs_val: List[float]
    # input_token_logprobs_idx: List[int]
    # output_token_logprobs_val: List[float]
    # output_token_logprobs_idx: List[int]
    # input_top_logprobs_val: List[List]
    # input_top_logprobs_idx: List[List]
    # output_top_logprobs_val: List[List]
    # output_top_logprobs_idx: List[List]

@dataclass
class GenerateReqInput:
    # The input prompt. It can be a single prompt or a batch of prompts.
    text: Optional[Union[List[str], str]] = None
    # The token ids for text; one can specify either text or input_ids
    # input_ids: Optional[Union[List[List[int]], List[int]]] = None
    # The embeddings for input_ids; one can specify either text or input_ids or input_embeds.
    # input_embeds: Optional[Union[List[List[List[float]]], List[List[float]]]] = None
    # The image input. It can be a file name, a url, or base64 encoded string.
    # See also python/sglang/srt/utils.py:load_image.
    # image_data: Optional[Union[List[str], str]] = None
    # The sampling_params. See descriptions below.
    sampling_params: Optional[Union[List[Dict], Dict]] = None
    # The request id.
    rid: Optional[Union[List[str], str]] = None
    # Whether to return logprobs.
    # return_logprob: Optional[Union[List[bool], bool]] = None
    # If return logprobs, the start location in the prompt for returning logprobs.
    # By default, this value is "-1", which means it will only return logprobs for output tokens.
    # logprob_start_len: Optional[Union[List[int], int]] = None
    # If return logprobs, the number of top logprobs to return at each position.
    # top_logprobs_num: Optional[Union[List[int], int]] = None
    # Whether to detokenize tokens in text in the returned logprobs.
    # return_text_in_logprobs: bool = False # Changed to commented as per implied suggestion structure
    # Whether to stream output.
    stream: bool = False
    # Whether to log metrics for this request (e.g. health_generate calls do not log metrics)
    # log_metrics: bool = True

    # The modalities of the image data [image, multi-images, video]
    # modalities: Optional[List[str]] = None
    # LoRA related
    # lora_path: Optional[Union[List[Optional[str]], Optional[str]]] = None

    # Session info for continual prompting
    # session_params: Optional[Union[List[Dict], Dict]] = None
    # Custom logit processor (serialized function)
    # custom_logit_processor: Optional[Union[List[Optional[str]], Optional[str]]] = None

    def normalize_batch_and_arguments(self):
        if self.text is None:
            raise ValueError(
                "text should be provided."
            )

        # Derive the original batch_size and is_single state
        if isinstance(self.text, str):
            self.is_single = True
            self.batch_size = 1  # Original batch size
        else:
            self.is_single = False
            self.batch_size = len(self.text)  # Original batch size

        # Determine parallel_sample_num from sampling_params
        if self.sampling_params is None:
            self.parallel_sample_num = 1
        elif isinstance(self.sampling_params, dict):
            self.parallel_sample_num = self.sampling_params.get("n", 1)
        else:  # isinstance(self.sampling_params, list)
            self.parallel_sample_num = self.sampling_params[0].get("n", 1)
            assert all(
                self.parallel_sample_num == sp.get("n", 1)
                for sp in self.sampling_params
            ), "The parallel_sample_num should be the same for all samples in sample params."

        # Expand self.text and update self.is_single based on parallel_sample_num
        if self.parallel_sample_num > 1:
            if self.is_single: # Original input was single
                self.text = [self.text] * self.parallel_sample_num
            else: # Original input was a batch
                self.text = [item for item in self.text for _ in range(self.parallel_sample_num)]
            self.is_single = False # Now effectively a batch due to parallel sampling

        # Final number of requests
        num = len(self.text)

        # Fill in default arguments
        if self.is_single: # This path is taken if original was single AND parallel_sample_num is 1
            if self.sampling_params is None:
                self.sampling_params = {}
            if self.rid is None:
                self.rid = uuid.uuid4().hex
            # Other commented-out parameters' defaults are removed
        else: # Batch mode (original batch, or single expanded by parallel sampling)
            # sampling_params
            if self.sampling_params is None:
                self.sampling_params = [{}] * num
            elif not isinstance(self.sampling_params, list): # Single dict provided
                self.sampling_params = [self.sampling_params] * num
            else: # List provided
                if len(self.sampling_params) == self.batch_size and self.parallel_sample_num > 1:
                    self.sampling_params = [p for p in self.sampling_params for _ in range(self.parallel_sample_num)]
                assert len(self.sampling_params) == num, \
                    f"sampling_params length mismatch. Expected {num}, got {len(self.sampling_params)}"

            # rid
            if self.rid is None:
                self.rid = [uuid.uuid4().hex for _ in range(num)]
            elif not isinstance(self.rid, list): # Single string provided
                self.rid = [self.rid] * num
            else: # List provided
                if len(self.rid) == self.batch_size and self.parallel_sample_num > 1:
                    self.rid = [r for r in self.rid for _ in range(self.parallel_sample_num)]
                assert len(self.rid) == num, \
                    f"rid length mismatch. Expected {num}, got {len(self.rid)}"

            # IGNORE: Removed handling for image_data, return_logprob, logprob_start_len,
            # top_logprobs_num, custom_logit_processor as they are commented out.

    def regenerate_rid(self):
        import uuid # Add import here
        self.rid = uuid.uuid4().hex
        return self.rid

    def __getitem__(self, i):
        return GenerateReqInput(
            text=self.text[i] if self.text is not None else None,
            input_ids=self.input_ids[i] if self.input_ids is not None else None,
            image_data=self.image_data[i],
            sampling_params=self.sampling_params[i],
            rid=self.rid[i],
            return_logprob=self.return_logprob[i],
            logprob_start_len=self.logprob_start_len[i],
            top_logprobs_num=self.top_logprobs_num[i],
            return_text_in_logprobs=self.return_text_in_logprobs,
            stream=self.stream,
            log_metrics=self.log_metrics,
            modalities=self.modalities[i] if self.modalities else None,
            lora_path=self.lora_path[i] if self.lora_path is not None else None,
            custom_logit_processor=(
                self.custom_logit_processor[i]
                if self.custom_logit_processor is not None
                else None
            ),
        )
    
@dataclass
class TokenizedGenerateReqInput:
    # The request id
    rid: str
    # The input text
    input_text: str
    # The input token ids
    input_ids: List[int]
    # The image inputs
    # image_inputs: dict
    # The sampling parameters
    sampling_params: SamplingParams
    # Whether to return the logprobs
    # return_logprob: bool
    # If return logprobs, the start location in the prompt for returning logprobs.
    # logprob_start_len: int
    # If return logprobs, the number of top logprobs to return at each position.
    # top_logprobs_num: int
    # Whether to stream output
    stream: bool

    # LoRA related
    # lora_path: Optional[str] = None  # None means just use the base model
    # The input embeds
    # input_embeds: Optional[Union[List[List[List[float]]], List[List[float]]]] = None

    # Session info for continual prompting
    # session_params: Optional[SessionParams] = None

    # Custom logit processor (serialized function)
    # TODO (hpguo): Add an example and update doc string here
    # custom_logit_processor: Optional[str] = None