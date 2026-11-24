import torch

from agsm.config import build_model, load_config
from training.losses import compute_losses
from training.seed import seed_everything


def main():
    torch.set_num_threads(1)
    seed_everything(42)
    config = load_config("configs/training/smoke.yaml")
    model = build_model(config)
    tokens = torch.randint(0, config["model"]["vocab_size"], (2, 8))
    output = model(tokens)
    loss = compute_losses(output, tokens.roll(-1, 1), config["loss"])["total"]
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    print(
        f"CPU forward/backward passed. Loss {loss.item():.4f}, state {output.state.nbytes} bytes."
    )


if __name__ == "__main__":
    main()
