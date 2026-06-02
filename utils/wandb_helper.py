import os
import sys

try:
    import wandb
except ImportError:
    wandb = None

def init_wandb(
    mode: str = "disabled",
    project: str = "machine-unlearning-benchmark",
    name: str = None,
    group: str = None,
    job_type: str = None,
    config: dict = None,
):
    """
    Initializes wandb if mode is "online" or "offline".
    Returns the run object or None.
    """
    if wandb is None:
        if mode in ["online", "offline"]:
            print("[WARN] wandb is not installed. Ignoring wandb initialization.")
        return None

    if mode in ["online", "offline"]:
        try:
            return wandb.init(
                project=project,
                name=name,
                group=group,
                job_type=job_type,
                config=config,
                mode=mode,
            )
        except Exception as e:
            print(f"[WARN] Failed to initialize wandb: {e}")
            return None
    return None

def log_wandb(metrics: dict, step: int = None):
    """
    Logs metrics dict to active wandb run if available.
    """
    if wandb is not None and wandb.run is not None:
        try:
            wandb.log(metrics, step=step)
        except Exception as e:
            print(f"[WARN] Failed to log to wandb: {e}")
