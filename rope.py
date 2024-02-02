from typing import Tuple
import torch


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)


def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    _, seqlen, _, _ = query.shape
    device = query.device
    # todo
    #
    # Please refer to slide 22 in https://phontron.com/class/anlp2024/assets/slides/anlp-05-transformers.pdf.
    # You may also benefit from https://blog.eleuther.ai/rotary-embeddings/.

    # reshape xq and xk to match the complex representation
    query_real, query_imag = (
        query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    )
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)

    # First, compute the trigonometric values in the second and fourth columns in
    # slide 22 (linked above).
    theta = 1.0 / torch.pow(theta, torch.arange(0, head_dim, 2).float() / head_dim)
    mtheta = torch.outer(torch.arange(seqlen).float(), theta)
    cos = reshape_for_broadcast(torch.cos(mtheta), query_real)
    sin = reshape_for_broadcast(torch.sin(mtheta), query_real)

    # Then, combine these trigonometric values with the tensors query_real, query_imag,
    # key_real, and key_imag.
    query_odd = query_real * cos - query_imag * sin
    query_even = query_real * sin + query_imag * cos
    key_odd = key_real * cos - key_imag * sin
    key_even = key_real * sin + key_imag * cos

    query_out = (
        torch.stack((query_odd, query_even), dim=-1).reshape(query.shape).to(device)
    )
    key_out = torch.stack((key_odd, key_even), dim=-1).reshape(key.shape).to(device)
    # Return the rotary position embeddings for the query and key tensors
    return query_out, key_out
