from .fla_adapter import FLABaseline


class GatedDeltaNetAdapter(FLABaseline):
    """Gated DeltaNet via the upstream Flash Linear Attention implementation."""

    def __init__(self, vocab_size, model_dim, output_dim=None, **_):
        if model_dim % 32:
            raise ValueError("Gated DeltaNet model_dim must be divisible by 32")
        from fla.models.gated_deltanet import GatedDeltaNetConfig, GatedDeltaNetModel

        config = GatedDeltaNetConfig(
            vocab_size=vocab_size,
            hidden_size=model_dim,
            head_dim=32,
            num_heads=model_dim // 32,
            num_hidden_layers=2,
        )
        super().__init__(GatedDeltaNetModel(config), model_dim, vocab_size, output_dim)
