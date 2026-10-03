"""Small no-op wandb fallback for local-only runs."""
import os
import tempfile


class _Run:
    def __init__(self, run_dir=None):
        self.id = "local"
        self.dir = run_dir or tempfile.mkdtemp(prefix="wandb-disabled-")


class _Config(dict):
    def update(self, values=None, allow_val_change=None, **kwargs):
        if values:
            super().update(values)
        super().update(kwargs)


run = None
config = _Config()


def init(*args, **kwargs):
    global run
    run_dir = kwargs.get("dir")
    if run_dir:
        run_dir = os.path.join(run_dir, "files")
        os.makedirs(run_dir, exist_ok=True)
    run = _Run(run_dir)
    return run


def log(*args, **kwargs):
    return None


def save(*args, **kwargs):
    return None


def finish(*args, **kwargs):
    global run
    run = None


class Api:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "wandb is not installed. Use --use_local_wandb for local artifacts, "
            "or install wandb for remote API access."
        )
