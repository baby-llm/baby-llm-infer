import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import logger

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA device.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device (Apple Silicon).")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device.")
    return device

def load_model_and_tokenizer(model_name="gpt2", device=None):
    if device is None:
        device = get_device()

    logger.info(f"Loading model: {model_name} onto device: {device}")
    try:
        dtype = torch.float16 if device.type in ['cuda', 'mps'] else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
        if device.type == 'cpu':
             model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)

    except Exception as e:
        logger.warning(f"Could not load model with float16, trying float32: {e}")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set padding token if it doesn't exist (for batching)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    logger.info(f"Model {model_name} loaded successfully.")
    return model, tokenizer, device