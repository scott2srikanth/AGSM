import torch


class MetricAccumulator:
    """Aggregate task metrics over labeled positions and diagnostics over tokens."""

    def __init__(self):
        self.totals = {}
        self.counts = {}

    def add(self, name, value, count):
        self.totals[name] = self.totals.get(name, 0.0) + float(value) * count
        self.counts[name] = self.counts.get(name, 0) + count

    @torch.no_grad()
    def update(self, output, targets, losses, task):
        mask = targets.ne(-100)
        labeled = int(mask.sum())
        tokens = targets.numel()
        for name, loss in losses.items():
            self.add(f"loss_{name}", loss, labeled if name == "task" else tokens)
        if task != "regression" and labeled:
            correct = ((output.logits.argmax(-1) == targets) & mask).sum()
            self.add("accuracy", correct / labeled, labeled)
        for name, value in output.diagnostics.items():
            self.add(name, value.float().mean(), tokens)

    def result(self):
        return {
            name: value / self.counts[name]
            for name, value in self.totals.items()
            if self.counts[name]
        }
