from torch import nn

from agsm.model import ModelOutput


class FLABaseline(nn.Module):
    """Task-head wrapper around a verified Flash Linear Attention backbone."""

    def __init__(self, backbone, model_dim, vocab_size, output_dim):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(model_dim, output_dim or vocab_size)

    def forward(self, tokens, state=None, mask=None):
        if tokens.device.type != "cuda":
            raise RuntimeError("This FLA adapter requires a supported CUDA runtime")
        if self.training and state is not None:
            raise ValueError("FLA streaming cache is inference-only")
        result = self.backbone(
            input_ids=tokens,
            attention_mask=mask,
            past_key_values=state,
            use_cache=not self.training,
            return_dict=True,
        )
        return ModelOutput(
            self.head(result.last_hidden_state), result.past_key_values, {}
        )
