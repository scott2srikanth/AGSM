import subprocess


def git_metadata():
    def git(*args):
        result = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    revision = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {"commit": revision, "dirty": None if status is None else bool(status)}


class ExperimentLogger:
    """Optional W&B transport. Checkpointed metric history is always local."""

    def __init__(self, settings, metadata, resume):
        self.run = None
        if settings.get("backend") == "wandb":
            import wandb

            experiment_id = metadata["experiment_id"]
            self.run = wandb.init(
                project=settings["project"],
                config=metadata,
                id=experiment_id,
                resume="allow" if resume else None,
                mode=settings.get("mode"),
            )

    def log(self, row):
        if self.run is not None:
            self.run.log(
                {key: val for key, val in row.items() if key != "epoch"},
                step=row["epoch"],
            )

    def close(self):
        if self.run is not None:
            self.run.finish()
