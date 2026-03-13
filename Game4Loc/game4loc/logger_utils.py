import logging
import re
import time
from contextlib import ContextDecorator
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _sanitize_name(name: str) -> str:
    sanitized = re.sub(r"[^0-9a-zA-Z_-]+", "_", name.strip())
    return sanitized or "algorithm"


def _project_root() -> Path:
    # game4loc/logger_utils.py -> Game4Loc/
    return Path(__file__).resolve().parents[1]


def _build_log_file_path(algorithm_name: str) -> Path:
    log_dir = _project_root() / "Log"
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    base_name = f"{_sanitize_name(algorithm_name)}_{ts}"
    candidate = log_dir / f"{base_name}.log"

    # Keep requested base format and append index only on collisions.
    index = 1
    while candidate.exists():
        candidate = log_dir / f"{base_name}_{index:02d}.log"
        index += 1
    return candidate


def setup_logger(
    algorithm_name: str,
    log_level: int = logging.DEBUG,
    logger_name: str = "game4loc",
) -> tuple[logging.Logger, str]:
    log_file = _build_log_file_path(algorithm_name)
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.propagate = False

    # Ensure every run creates a clean logger/file handler.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("日志系统初始化完成，日志文件: %s", log_file)
    return logger, str(log_file)


def log_config(logger: logging.Logger, config: Any) -> None:
    if hasattr(config, "__dict__"):
        items = sorted(config.__dict__.items())
    elif isinstance(config, dict):
        items = sorted(config.items())
    else:
        logger.info("配置: %s", str(config))
        return

    logger.info("实验配置:")
    for key, value in items:
        logger.info("  %s = %s", key, value)


class log_timer(ContextDecorator):
    def __init__(
        self,
        logger: logging.Logger,
        step_name: str,
        level: int = logging.INFO,
        sync_cuda: bool = False,
    ):
        self.logger = logger
        self.step_name = step_name
        self.level = level
        self.sync_cuda = sync_cuda
        self.start = 0.0

    def _maybe_sync_cuda(self) -> None:
        if self.sync_cuda and torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()

    def __enter__(self):
        self._maybe_sync_cuda()
        self.start = time.perf_counter()
        self.logger.log(self.level, "[计时器][开始] %s", self.step_name)
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        self._maybe_sync_cuda()
        elapsed = time.perf_counter() - self.start
        if exc_type is None:
            self.logger.log(self.level, "[计时器][结束] %s | 耗时=%.6fs", self.step_name, elapsed)
        else:
            self.logger.exception(
                "[计时器][异常] %s 在 %.6fs 后失败", self.step_name, elapsed
            )
        return False


def timed(step_name: str | None = None, level: int = logging.DEBUG, sync_cuda: bool = False):
    def decorator(func):
        resolved_name = step_name or func.__name__

        def wrapper(*args, **kwargs):
            logger = kwargs.get("logger")
            if logger is None and len(args) > 0 and hasattr(args[0], "logger"):
                logger = getattr(args[0], "logger")

            if logger is None:
                return func(*args, **kwargs)

            with log_timer(
                logger=logger,
                step_name=resolved_name,
                level=level,
                sync_cuda=sync_cuda,
            ):
                return func(*args, **kwargs)

        return wrapper

    return decorator
