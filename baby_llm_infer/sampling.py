import torch
import torch.nn.functional as F
from typing import Optional

@torch.inference_mode()
def sample_logits(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
    """
    Samples token IDs from logits using temperature, top-p, and top-k.
    Args:
        logits: Tensor of shape (batch_size, vocab_size)
        temperature: Softmax temperature. Lower -> more deterministic.
        top_p: Nucleus sampling probability threshold.
        top_k: Keep only top_k most likely tokens.
    Returns:
        Tensor of shape (batch_size,) with sampled token IDs.
    """
    if temperature == 0: # Treat 0 temperature as greedy (argmax)
        return torch.argmax(logits, dim=-1)

    logits = logits / temperature

    # Apply Top-K
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))  # Safety check
        # Remove tokens with likelihood less than the k-th token
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, float('-inf'))

    # Apply Top-P (Nucleus Sampling)
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))

    # Sample from the modified distribution
    probs = F.softmax(logits, dim=-1)
    # Handle potential numerical issues where probs might sum to slightly less than 1
    # or if all logits were -inf (shouldn't happen with valid inputs but good practice)
    probs = torch.nan_to_num(probs, nan=0.0)
    # Ensure probabilities sum to 1 if possible, otherwise re-normalize if needed after filtering
    # This simple multinomial sample handles cases where probs might not sum exactly to 1 after filtering
    # by effectively re-normalizing internally if needed.
    # Ensure there's at least one valid token to sample from. If not (all probs are 0),
    # it might indicate an issue or the need for a fallback (e.g., EOS token).
    # For simplicity, we assume sampling is always possible here.
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

    return next_tokens