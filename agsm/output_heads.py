from torch import nn


class ClassificationHead(nn.Linear):
    """Per-token class logits. Choose supervised positions using target masks."""


class RetrievalHead(nn.Linear):
    """Per-token logits over retrieved symbols."""


class RegressionHead(nn.Linear):
    """Per-token continuous predictions."""


class LanguageModelHead(nn.Linear):
    """Per-token next-token logits."""
