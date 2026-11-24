from torch.optim import AdamW


def make_optimizer(model, settings):
    return AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
