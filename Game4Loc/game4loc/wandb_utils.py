import time
from contextlib import ContextDecorator
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


def _project_root() -> Path:
    # game4loc/wandb_utils.py -> Game4Loc/
    return Path(__file__).resolve().parents[1]


def get_wandb_dir() -> str:
    # Required local storage directory: Game4Loc/W&B/
    local_dir = _project_root() / "W&B"
    local_dir.mkdir(parents=True, exist_ok=True)
    return str(local_dir)


def init_wandb_run(config: Any, algorithm_name: str, logger=None):
    if wandb is None:
        raise ImportError("wandb is not installed. Please install with: pip install wandb")

    cfg = dict(vars(config)) if hasattr(config, "__dict__") else dict(config)
    run_name = f"{algorithm_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run = wandb.init(
        project="Game4Loc-Experiment",
        name=run_name,
        dir=get_wandb_dir(),
        config=cfg,
        reinit=True,
    )
    if logger is not None:
        logger.info("W&B 初始化完成: project=%s, run=%s, dir=%s", "Game4Loc-Experiment", run_name, get_wandb_dir())
    return run


def safe_log(run, metrics: dict, step: int | None = None, commit: bool = True):
    if run is None:
        return
    if step is None:
        run.log(metrics, commit=commit)
    else:
        run.log(metrics, step=step, commit=commit)


def finish_wandb(run, logger=None):
    if run is None:
        return
    run.finish()
    if logger is not None:
        logger.info("W&B 实验记录已结束")


class WandbStepTimer(ContextDecorator):
    def __init__(self, step_name: str, logger=None, run=None, step: int | None = None, sync_cuda: bool = False):
        self.step_name = step_name
        self.logger = logger
        self.run = run
        self.step = step
        self.sync_cuda = sync_cuda
        self.start = 0.0

    def _maybe_sync_cuda(self):
        try:
            import torch
            if self.sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

    def __enter__(self):
        self._maybe_sync_cuda()
        self.start = time.perf_counter()
        if self.logger is not None:
            self.logger.debug("[W&B计时][开始] %s", self.step_name)
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        self._maybe_sync_cuda()
        elapsed = time.perf_counter() - self.start
        safe_log(self.run, {f"time/{self.step_name}_s": elapsed}, step=self.step)
        if self.logger is not None:
            self.logger.debug("[W&B计时][结束] %s | 耗时=%.6fs", self.step_name, elapsed)
        return False
