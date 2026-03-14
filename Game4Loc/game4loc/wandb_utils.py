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


def _infer_dataset_name(config: Any) -> str:
    name = None
    if hasattr(config, "dataset"):
        name = getattr(config, "dataset")
    elif hasattr(config, "data_root"):
        p = str(getattr(config, "data_root"))
        if "GTA" in p or "GTA-UAV" in p:
            name = "GTA-UAV"
        elif "U1652" in p or "University" in p:
            name = "U1652"
        elif "VisLoc" in p:
            name = "VisLoc"
        elif "DenseUAV" in p:
            name = "DenseUAV"
        elif "SUES" in p or "SUESS" in p:
            name = "SUES"
    return str(name) if name else "Unknown"


def _infer_run_type(algorithm_name: str, config: Any) -> str:
    alg = str(algorithm_name).lower()
    if "eval" in alg:
        return "eval"
    # fallback: some configs may carry explicit flags
    if hasattr(config, "run_type"):
        return str(getattr(config, "run_type"))
    return "train"


def init_wandb_run(config: Any, algorithm_name: str, logger=None, dataset_name: str | None = None, run_type: str | None = None):
    if wandb is None:
        raise ImportError("wandb is not installed. Please install with: pip install wandb")

    cfg = dict(vars(config)) if hasattr(config, "__dict__") else dict(config)
    dataset = dataset_name or _infer_dataset_name(config)
    rtype = run_type or _infer_run_type(algorithm_name, config)
    with_match = getattr(config, "with_match", None)
    run_name = f"{algorithm_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run = wandb.init(
        project="Game4Loc-Experiment",
        name=run_name,
        dir=get_wandb_dir(),
        config=cfg,
        group=dataset,
        tags=[f"dataset:{dataset}", f"mode:{rtype}", f"model:{algorithm_name}"] + (
            [f"with_match:{bool(with_match)}"] if with_match is not None else []
        ),
        reinit=True,
    )
    try:
        notes = f"dataset={dataset}, mode={rtype}, model={algorithm_name}" + (
            f", with_match={bool(with_match)}" if with_match is not None else ""
        )
        run.notes = notes
    except Exception:
        pass
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

    @staticmethod
    def _canonical_key(step_name: str) -> str:
        s = step_name.lower()
        if "model_initialization" in s:
            return "time/model_init_s"
        if "data_loading" in s:
            return "time/data_loading_s"
        if "train_epoch" in s:
            return "time/train/epoch_s"
        if "evaluate_epoch" in s or "evaluation" in s:
            return "time/eval/overall_s"
        if "save_checkpoint" in s:
            return "time/train/save_checkpoint_s"
        # fallback: use last token
        token = step_name.split("/")[-1]
        return f"time/{token}_s"

    def __enter__(self):
        self._maybe_sync_cuda()
        self.start = time.perf_counter()
        if self.logger is not None:
            self.logger.debug("[W&B计时][开始] %s", self.step_name)
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        self._maybe_sync_cuda()
        elapsed = time.perf_counter() - self.start
        key = self._canonical_key(self.step_name)
        safe_log(self.run, {key: elapsed}, step=self.step)
        if self.logger is not None:
            self.logger.debug("[W&B计时][结束] %s | 耗时=%.6fs", self.step_name, elapsed)
        return False
