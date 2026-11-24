from torch.optim.lr_scheduler import CosineAnnealingLR


def make_scheduler(optimizer, settings):
    return CosineAnnealingLR(optimizer, T_max=settings["epochs"])
