"""
Attention visualization for the DNA Transformer encoder
(src/genomics/models.py::DNATransformerEncoder).

Produces a per-base attention score that can be rendered as a heatmap strip
under the raw sequence — analogous to Grad-CAM, but for the genomics branch.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch


def get_attention_scores(model: torch.nn.Module, input_tensor: torch.Tensor, device: str = "cpu") -> np.ndarray:
    """
    Args:
        model: trained DNATransformerEncoder in eval mode.
        input_tensor: (1, 4, L) one-hot encoded sequence tensor.

    Returns:
        (L,) numpy array of attention weights over sequence positions
        (the [CLS] token's own attention weight is dropped).
    """
    model.eval()
    with torch.no_grad():
        attn = model.get_attention(input_tensor.to(device))  # (1, L+1)
    attn = attn.squeeze(0).cpu().numpy()
    return attn[1:]  # drop [CLS]-to-[CLS] weight, keep per-base scores


def plot_attention_strip(sequence: str, attention_scores: np.ndarray, save_path: str = None):
    """
    Renders the DNA sequence as text with a heatmap strip beneath it showing
    per-base attention. Truncates very long sequences for readability.
    """
    max_display = 120
    seq_display = sequence[:max_display]
    scores_display = attention_scores[:max_display]
    scores_norm = (scores_display - scores_display.min()) / (scores_display.ptp() + 1e-8)

    fig, ax = plt.subplots(figsize=(min(20, len(seq_display) * 0.15), 2))
    ax.imshow(scores_norm[np.newaxis, :], cmap="Reds", aspect="auto")
    ax.set_xticks(range(len(seq_display)))
    ax.set_xticklabels(list(seq_display), fontsize=7)
    ax.set_yticks([])
    ax.set_title("Model attention over DNA sequence (darker = higher attention)")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig
