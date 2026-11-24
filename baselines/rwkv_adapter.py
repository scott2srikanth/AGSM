from .fla_adapter import FLABaseline


class RWKV7Adapter(FLABaseline):
    """RWKV-7 via the upstream Flash Linear Attention implementation."""

    def __init__(self, vocab_size, model_dim, output_dim=None, **_):
        if model_dim % 64:
            raise ValueError("RWKV-7 model_dim must be divisible by 64")
        from fla.models.rwkv7 import RWKV7Config, RWKV7Model

        config = RWKV7Config(
            vocab_size=vocab_size,
            hidden_size=model_dim,
            head_dim=64,
            num_heads=model_dim // 64,
            num_hidden_layers=2,
        )
        super().__init__(RWKV7Model(config), model_dim, vocab_size, output_dim)
