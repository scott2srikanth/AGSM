from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .controller import JointController
from .encoder import InputEncoder
from .fast_memory import FastAssociativeMemory
from .output_heads import (
    ClassificationHead,
    LanguageModelHead,
    RegressionHead,
    RetrievalHead,
)
from .refinement import AdaptiveRefinement
from .selective_gate import SelectiveInformationGate
from .slow_memory import SlowContextualMemory
from .state import RecurrentState


@dataclass
class ModelOutput:
    logits: Tensor
    state: RecurrentState | Tensor | None
    diagnostics: dict[str, Tensor]


class AGSM(nn.Module):
    """Attention-free causal model with functional, caller-owned recurrent state."""

    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        key_dim: int,
        value_dim: int,
        slow_state_dim: int,
        controller_dim: int,
        refinement_hidden_dim: int,
        max_refinement_steps: int,
        embedding_dim: int | None = None,
        dropout: float = 0.0,
        task: str = "retrieval",
        output_dim: int | None = None,
        memory: dict | None = None,
        controller: dict | None = None,
        inference_threshold: float = 0.99,
    ):
        super().__init__()
        memory = dict(memory or {})
        controller = dict(controller or {})
        self.memory_type = memory.get("type", "dual")
        self.use_residual = controller.get("residual_signal", True)
        self.use_memory_norm = controller.get("memory_norm_signal", True)
        self.adaptive_halting = controller.get("adaptive_halting", True)
        self.inference_threshold = inference_threshold
        self.encoder = InputEncoder(vocab_size, embedding_dim or model_dim, model_dim)
        self.selective = SelectiveInformationGate(
            model_dim, slow_state_dim, key_dim, value_dim
        )
        self.fast = FastAssociativeMemory(
            value_dim,
            key_dim,
            memory.get("update_precision", "float32"),
            memory.get("norm_limit", 16.0)
            if memory.get("norm_clipping", True)
            else None,
            memory.get("normalize_keys", True),
        )
        self.slow = SlowContextualMemory(model_dim, value_dim, slow_state_dim)
        self.controller = JointController(
            model_dim + slow_state_dim + 2 * value_dim + 2,
            controller_dim,
            model_dim,
            {
                gate: controller.get(f"adaptive_{gate}", True)
                for gate in JointController.DEFAULTS
            },
        )
        self.refinement = AdaptiveRefinement(
            model_dim,
            model_dim + slow_state_dim + value_dim,
            refinement_hidden_dim,
            max_refinement_steps,
            dropout,
        )
        heads = {
            "classification": ClassificationHead,
            "retrieval": RetrievalHead,
            "regression": RegressionHead,
            "language_model": LanguageModelHead,
        }
        self.head = heads[task](model_dim, output_dim or vocab_size)

    def initialize_state(self, batch_size: int, device=None) -> RecurrentState:
        weights = self.encoder.embedding.weight
        device = weights.device if device is None else device
        return RecurrentState(
            self.fast.initial_state(batch_size, device),
            self.slow.initial_state(batch_size, device, weights.dtype),
        )

    def reset_state(self, state: RecurrentState) -> RecurrentState:
        return RecurrentState(
            self.fast.reset_state(state.fast), self.slow.reset_state(state.slow)
        )

    def step(
        self, x: Tensor, state: RecurrentState, valid: Tensor, compact: bool = True
    ) -> tuple[Tensor, RecurrentState, dict[str, Tensor]]:
        u, query, key, value = self.selective(x, state.slow)
        query, key = self.fast.address(query), self.fast.address(key)
        raw_read = self.fast.read(state.fast, query)
        residual = value.to(key.dtype) - self.fast.read(state.fast, key)
        rho = residual.norm(dim=-1) / value.to(key.dtype).norm(dim=-1).clamp_min(1e-6)
        previous_norm = self.fast.norm(state.fast)
        signals = (
            rho.log1p()
            if self.use_residual and self.memory_type != "slow"
            else torch.zeros_like(rho),
            previous_norm if self.use_memory_norm else torch.zeros_like(previous_norm),
        )
        read = raw_read.to(u.dtype)
        gates, context = self.controller(
            torch.cat(
                (
                    u,
                    state.slow,
                    read,
                    value,
                    *(signal[:, None].to(u.dtype) for signal in signals),
                ),
                dim=-1,
            )
        )
        strength = gates["write"] * gates["update_strength"]
        fast, slow = state.fast, state.slow
        if self.memory_type != "slow":
            fast = self.fast.write(fast, key, residual, gates["retention"], strength)
        else:
            strength = torch.zeros_like(strength)
        if self.memory_type != "fast":
            slow = self.slow(u, read, slow, gates["slow_update"])
        fast = torch.where(valid[:, None, None], fast, state.fast)
        slow = torch.where(valid[:, None], slow, state.slow)
        features = torch.cat((u, slow, read * gates["retrieval"].to(u.dtype)), dim=-1)
        result = self.refinement(
            features,
            context,
            self.controller.halting_probability,
            self.inference_threshold,
            self.adaptive_halting,
            compact,
        )
        logits = self.head(result.representation)
        diagnostics = {
            "expected_steps": result.expected,
            "executed_steps": result.executed,
            "write_strength": strength.squeeze(-1),
            "residual": rho,
            "fast_norm": self.fast.norm(fast),
            "slow_norm": slow.float().norm(dim=-1),
            **{name: gate.squeeze(-1) for name, gate in gates.items()},
        }
        return logits, RecurrentState(fast, slow), diagnostics

    def forward(
        self,
        tokens: Tensor,
        state: RecurrentState | None = None,
        mask: Tensor | None = None,
        *,
        compact: bool = True,
    ) -> ModelOutput:
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError("tokens must have shape [batch, nonempty sequence]")
        if state is None:
            state = self.initialize_state(tokens.shape[0], tokens.device)
        if mask is None:
            mask = torch.ones_like(tokens, dtype=torch.bool)
        if mask.shape != tokens.shape or mask.dtype != torch.bool:
            raise ValueError("mask must be boolean with the same shape as tokens")
        if state.fast.shape != (
            tokens.shape[0],
            self.fast.value_dim,
            self.fast.key_dim,
        ):
            raise ValueError("fast state shape is incompatible with tokens and model")
        if state.slow.shape != (tokens.shape[0], self.slow.slow_dim):
            raise ValueError("slow state shape is incompatible with tokens and model")
        outputs = []
        metrics = []
        for x, valid in zip(self.encoder(tokens).unbind(1), mask.unbind(1)):
            logits, state, diagnostics = self.step(x, state, valid, compact)
            outputs.append(logits)
            metrics.append(diagnostics)
        return ModelOutput(
            torch.stack(outputs, dim=1),
            state,
            {
                name: torch.stack([step[name] for step in metrics], dim=1)
                for name in metrics[0]
            },
        )

    def process_chunk(
        self,
        tokens: Tensor,
        state: RecurrentState | None = None,
        mask: Tensor | None = None,
    ) -> ModelOutput:
        return self(tokens, state, mask)

    @staticmethod
    def get_state(output: ModelOutput) -> RecurrentState | None:
        return output.state
