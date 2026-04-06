import os
import time
import shutil
import logging
import atexit
import torch
import argparse
from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import (
    get_constant_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from game4loc.dataset.visloc import VisLocDatasetEval, VisLocDatasetTrain, get_transforms
from game4loc.utils import setup_system
from game4loc.logger_utils import setup_logger, log_config, log_timer, log_run_header
from game4loc.wandb_utils import init_wandb_run, finish_wandb, WandbStepTimer, safe_log
from game4loc.trainer.trainer import train_with_weight
from game4loc.evaluate.visloc import evaluate
from game4loc.loss import WeightedInfoNCE
from game4loc.models.model import DesModel


def parse_tuple(s):
    try:
        return tuple(map(int, s.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Tuple must be integers separated by commas") from exc


@dataclass
class Configuration:
    model: str = "vit_base_patch16_rope_reg1_gap_256.sbb_in1k"
    img_size: int = 384
    share_weights: bool = True

    mixed_precision: bool = True
    custom_sampling: bool = True
    seed: int = 1
    epochs: int = 10
    batch_size: int = 40
    batch_size_eval: int = 128
    verbose: bool = False
    gpu_ids: tuple = (0, 1)

    train_ratio: float = 1.0
    eval_every_n_epoch: int = 1
    normalize_features: bool = True
    eval_gallery_n: int = -1

    clip_grad = 100.0
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False

    label_smoothing: float = 0.0
    k: float = 5.0
    lr: float = 0.001
    scheduler: str = "cosine"
    warmup_epochs: float = 0.1
    lr_end: float = 0.0001
    prob_flip: float = 0.5

    model_path: str = "./work_dir/visloc"
    use_wandb: bool = True

    train_mode: str = "pos_semipos"
    test_mode: str = "pos"
    zero_shot: bool = True
    checkpoint_start = None

    use_pose_attention: bool = False
    rotate_query_to_north: bool = False
    train_random_rotate90: bool = False
    pose_attn_floor: float = 0.3
    pose_gate_mode: str = "multiplicative"
    pose_gate_lambda: float = 0.5
    pose_gate_insert_stage: str = "pre_blocks"
    pose_fov_deg: float = 36.0

    num_workers: int = 0 if os.name == "nt" else 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False

    data_root: str = "./data/UAV_VisLoc_dataset"
    train_pairs_meta_file: str = "same-area-drone2sate-train-pose.json"
    test_pairs_meta_file: str = "same-area-drone2sate-test-pose.json"
    sate_img_dir: str = "satellite"


def build_scheduler(config, optimizer, train_dataloader, logger):
    train_steps = len(train_dataloader) * config.epochs
    warmup_steps = int(len(train_dataloader) * config.warmup_epochs)

    if config.scheduler == "polynomial":
        logger.info("学习率调度器: polynomial - 最大LR: %s - 结束LR: %s", config.lr, config.lr_end)
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_training_steps=train_steps,
            lr_end=config.lr_end,
            power=1.5,
            num_warmup_steps=warmup_steps,
        )
    elif config.scheduler == "cosine":
        logger.info("学习率调度器: cosine - 最大LR: %s", config.lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_training_steps=train_steps,
            num_warmup_steps=warmup_steps,
        )
    elif config.scheduler == "constant":
        logger.info("学习率调度器: constant - 最大LR: %s", config.lr)
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
        )
    else:
        scheduler = None

    logger.info("Warmup 步数: %s", warmup_steps)
    logger.info("训练总轮数: %s - 训练总步数: %s", config.epochs, train_steps)
    return scheduler


def train_script(config):
    dataset_name = "UAV-VisLoc"
    logger, log_path = setup_logger(
        algorithm_name=config.model,
        log_level=logging.DEBUG,
        run_type="train",
        dataset_name=dataset_name,
        log_dir=config.log_path,
    )
    wandb_run = None
    log_run_header(logger, run_mode="train", algorithm_name=config.model)

    save_time = time.strftime("%m%d%H%M%S")
    model_path = f"{config.model_path}/{config.model}/{save_time}"
    os.makedirs(model_path, exist_ok=True)
    shutil.copyfile(os.path.abspath(__file__), f"{model_path}/train.py")

    setup_system(
        seed=config.seed,
        cudnn_benchmark=config.cudnn_benchmark,
        cudnn_deterministic=config.cudnn_deterministic,
    )

    logger.info("训练输出目录: %s", model_path)
    logger.info("自动日志路径: %s", log_path)
    logger.info("训练起始检查点: %s", config.checkpoint_start)
    log_config(logger, config)

    if config.use_wandb:
        wandb_run = init_wandb_run(
            config=config,
            algorithm_name=config.model,
            logger=logger,
            dataset_name=dataset_name,
            run_type="train",
        )
        wandb_run.config.update(
            {
                "learning_rate": config.lr,
                "batch_size": config.batch_size,
            },
            allow_val_change=True,
        )
        atexit.register(finish_wandb, wandb_run, logger)
        safe_log(
            wandb_run,
            {
                "meta/log_path": log_path,
                "meta/model_path": model_path,
            },
        )

    logger.info("模型: %s", config.model)
    logger.info("Use pose attention: %s", config.use_pose_attention)
    logger.info("Rotate UAV query to north by Phi1: %s", config.rotate_query_to_north)
    logger.info("Train random rotate90 augmentation: %s", config.train_random_rotate90)

    with log_timer(logger, "model_initialization", level=logging.INFO, sync_cuda=True), \
         WandbStepTimer("model_initialization", logger=logger, run=wandb_run, sync_cuda=True):
        model = DesModel(
            model_name=config.model,
            pretrained=True,
            img_size=config.img_size,
            share_weights=config.share_weights,
        )
        model.use_pose_attention = config.use_pose_attention
        model.pose_attn_floor = config.pose_attn_floor
        model.pose_gate_mode = config.pose_gate_mode
        model.pose_gate_lambda = config.pose_gate_lambda
        model.pose_gate_insert_stage = config.pose_gate_insert_stage
        model.fov_deg = config.pose_fov_deg

    data_config = model.get_config()
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = (config.img_size, config.img_size)

    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)

    if config.checkpoint_start is not None:
        with log_timer(logger, "load_checkpoint", level=logging.INFO, sync_cuda=True):
            logger.info("加载检查点: %s", config.checkpoint_start)
            model_state_dict = torch.load(config.checkpoint_start)
            model.load_state_dict(model_state_dict, strict=False)

    logger.info("可用 GPU 数量: %s", torch.cuda.device_count())
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)

    model = model.to(config.device)

    logger.info("查询图像尺寸: %s", img_size)
    logger.info("图库图像尺寸: %s", img_size)
    logger.info("Mean: %s", mean)
    logger.info("Std: %s", std)

    with log_timer(logger, "data_loading", level=logging.INFO), \
         WandbStepTimer("data_loading", logger=logger, run=wandb_run):
        val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(
            img_size,
            mean=mean,
            std=std,
            random_rotate90=config.train_random_rotate90,
        )

        train_dataset = VisLocDatasetTrain(
            data_root=config.data_root,
            pairs_meta_file=config.train_pairs_meta_file,
            transforms_query=train_drone_transforms,
            transforms_gallery=train_sat_transforms,
            prob_flip=config.prob_flip,
            shuffle_batch_size=config.batch_size,
            mode=config.train_mode,
            train_ratio=config.train_ratio,
            use_pose=config.use_pose_attention,
            rotate_aerial_to_north=config.rotate_query_to_north,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=not config.custom_sampling,
            pin_memory=True,
        )

        query_dataset_test = VisLocDatasetEval(
            data_root=config.data_root,
            pairs_meta_file=config.test_pairs_meta_file,
            view="drone",
            mode=config.test_mode,
            transforms=val_transforms,
            use_pose=config.use_pose_attention,
            rotate_aerial_to_north=config.rotate_query_to_north,
        )
        query_img_list = query_dataset_test.images_name
        pairs_drone2sate_dict = query_dataset_test.pairs_drone2sate_dict
        query_center_loc_xy_list = query_dataset_test.images_center_loc_xy
        query_dataloader_test = DataLoader(
            query_dataset_test,
            batch_size=config.batch_size_eval,
            num_workers=config.num_workers,
            shuffle=False,
            pin_memory=True,
        )

        gallery_dataset_test = VisLocDatasetEval(
            data_root=config.data_root,
            pairs_meta_file=config.test_pairs_meta_file,
            view="sate",
            transforms=val_transforms,
            sate_img_dir=config.sate_img_dir,
        )
        gallery_img_list = gallery_dataset_test.images_name
        gallery_center_loc_xy_list = gallery_dataset_test.images_center_loc_xy
        gallery_topleft_loc_xy_list = gallery_dataset_test.images_topleft_loc_xy
        gallery_dataloader_test = DataLoader(
            gallery_dataset_test,
            batch_size=config.batch_size_eval,
            num_workers=config.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    logger.info("测试查询集图像数: %s", len(query_dataset_test))
    logger.info("测试图库图像数: %s", len(gallery_dataset_test))

    loss_function = WeightedInfoNCE(
        device=config.device,
        label_smoothing=config.label_smoothing,
        k=config.k,
    )
    scaler = GradScaler(init_scale=2.0 ** 10) if config.mixed_precision else None

    if config.decay_exclue_bias:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=config.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    scheduler = build_scheduler(config, optimizer, train_dataloader, logger)

    if config.zero_shot:
        logger.info("%s[Zero Shot]%s", 30 * "-", 30 * "-")
        with log_timer(logger, "zero_shot_evaluation", level=logging.INFO, sync_cuda=True), \
             WandbStepTimer("zero_shot_evaluation", logger=logger, run=wandb_run, sync_cuda=True):
            evaluate(
                config=config,
                model=model,
                query_loader=query_dataloader_test,
                gallery_loader=gallery_dataloader_test,
                query_list=query_img_list,
                gallery_list=gallery_img_list,
                query_center_loc_xy_list=query_center_loc_xy_list,
                gallery_center_loc_xy_list=gallery_center_loc_xy_list,
                gallery_topleft_loc_xy_list=gallery_topleft_loc_xy_list,
                pairs_dict=pairs_drone2sate_dict,
                ranks_list=[1, 5, 10],
                step_size=1000,
                cleanup=True,
                logger=logger,
                wandb_run=wandb_run,
                epoch=0,
            )

    best_score = 0.0
    for epoch in range(1, config.epochs + 1):
        logger.info("%s[Epoch: %s]%s", 30 * "-", epoch, 30 * "-")

        if config.custom_sampling:
            train_dataloader.dataset.shuffle()

        with log_timer(logger, f"train_epoch_{epoch}", level=logging.INFO, sync_cuda=True), \
             WandbStepTimer(f"train_epoch_{epoch}", logger=logger, run=wandb_run, step=epoch, sync_cuda=True):
            train_loss = train_with_weight(
                config,
                model,
                dataloader=train_dataloader,
                loss_function=loss_function,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                with_weight=True,
                logger=logger,
                wandb_run=wandb_run,
                epoch=epoch,
            )
            logger.info(
                "第 %s 轮: 训练损失 = %.3f, 学习率 = %.6f",
                epoch,
                train_loss,
                optimizer.param_groups[0]["lr"],
            )
            safe_log(
                wandb_run,
                {
                    "train/loss": train_loss,
                    "train/lr": optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )

        if (epoch % config.eval_every_n_epoch == 0 and epoch != 0) or epoch == config.epochs:
            logger.info("%s[Evaluate]%s", 30 * "-", 30 * "-")
            with log_timer(logger, f"evaluate_epoch_{epoch}", level=logging.INFO, sync_cuda=True), \
                 WandbStepTimer(f"evaluate_epoch_{epoch}", logger=logger, run=wandb_run, step=epoch, sync_cuda=True):
                r1_test = evaluate(
                    config=config,
                    model=model,
                    query_loader=query_dataloader_test,
                    gallery_loader=gallery_dataloader_test,
                    query_list=query_img_list,
                    gallery_list=gallery_img_list,
                    query_center_loc_xy_list=query_center_loc_xy_list,
                    gallery_center_loc_xy_list=gallery_center_loc_xy_list,
                    gallery_topleft_loc_xy_list=gallery_topleft_loc_xy_list,
                    pairs_dict=pairs_drone2sate_dict,
                    ranks_list=[1, 5, 10],
                    step_size=1000,
                    cleanup=True,
                    logger=logger,
                    wandb_run=wandb_run,
                    epoch=epoch,
                )

            if r1_test > best_score or epoch == config.epochs:
                best_score = r1_test
                safe_log(
                    wandb_run,
                    {
                        "eval/best_recall@1": best_score * 100.0,
                    },
                    step=epoch,
                )

                with log_timer(logger, f"save_checkpoint_epoch_{epoch}", level=logging.INFO, sync_cuda=True), \
                     WandbStepTimer(f"save_checkpoint_epoch_{epoch}", logger=logger, run=wandb_run, step=epoch, sync_cuda=True):
                    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                        torch.save(model.module.state_dict(), f"{model_path}/weights_e{epoch}_{r1_test:.4f}.pth")
                    else:
                        torch.save(model.state_dict(), f"{model_path}/weights_e{epoch}_{r1_test:.4f}.pth")

    with log_timer(logger, "save_final_checkpoint", level=logging.INFO, sync_cuda=True), \
         WandbStepTimer("save_final_checkpoint", logger=logger, run=wandb_run, sync_cuda=True):
        if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
            torch.save(model.module.state_dict(), f"{model_path}/weights_end.pth")
        else:
            torch.save(model.state_dict(), f"{model_path}/weights_end.pth")

    logger.info("训练完成，最佳 Recall@1=%.4f", best_score)
    safe_log(
        wandb_run,
        {
            "final/best_recall@1": best_score * 100.0,
        },
        step=config.epochs,
    )
    finish_wandb(wandb_run, logger=logger)


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for UAV-VisLoc with optional pose-driven prior.")
    parser.add_argument("--log_to_file", action="store_true", help="Log saving to file")
    parser.add_argument("--log_path", type=str, default=None, help="Log file path")
    parser.add_argument("--model_path", type=str, default="./work_dir/visloc", help="Checkpoint root directory")
    parser.add_argument("--data_root", type=str, default="./data/UAV_VisLoc_dataset", help="Data root")
    parser.add_argument("--train_pairs_meta_file", type=str, default="same-area-drone2sate-train-pose.json")
    parser.add_argument("--test_pairs_meta_file", type=str, default="same-area-drone2sate-test-pose.json")
    parser.add_argument("--model", type=str, default="vit_base_patch16_rope_reg1_gap_256.sbb_in1k")
    parser.add_argument("--no_share_weights", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eval_every_n_epoch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu_ids", type=parse_tuple, default=(0, 1))
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--batch_size_eval", type=int, default=128)
    parser.add_argument("--checkpoint_start", type=str, default=None)
    parser.add_argument("--train_mode", type=str, default="pos_semipos")
    parser.add_argument("--test_mode", type=str, default="pos")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--k", type=float, default=5.0)
    parser.add_argument("--train_ratio", type=float, default=1.0)
    parser.add_argument("--prob_flip", type=float, default=0.5)
    parser.add_argument("--use_pose_attention", action="store_true")
    parser.add_argument("--no_zero_shot", action="store_true")
    parser.add_argument("--rotate_query_to_north", action="store_true")
    parser.add_argument("--no_rotate_query_to_north", action="store_true")
    parser.add_argument("--train_random_rotate90", action="store_true")
    parser.add_argument("--pose_attn_floor", type=float, default=0.3)
    parser.add_argument("--pose_gate_mode", type=str, default="multiplicative", choices=["multiplicative", "residual"])
    parser.add_argument("--pose_gate_lambda", type=float, default=0.5)
    parser.add_argument(
        "--pose_gate_insert_stage",
        type=str,
        default="pre_blocks",
        choices=["pre_blocks", "after_block2", "after_block4"],
    )
    parser.add_argument("--pose_fov_deg", type=float, default=36.0)
    parser.add_argument("--no_custom_sampling", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = Configuration()
    config.data_root = args.data_root
    config.model_path = args.model_path
    config.train_pairs_meta_file = args.train_pairs_meta_file
    config.test_pairs_meta_file = args.test_pairs_meta_file
    config.log_to_file = args.log_to_file
    config.log_path = args.log_path

    if config.log_path is None:
        log_dir = "Log/train/"
    else:
        log_dir = config.log_path
    os.makedirs(log_dir, exist_ok=True)
    config.log_path = log_dir

    config.model = args.model
    config.share_weights = not args.no_share_weights
    config.epochs = args.epochs
    config.eval_every_n_epoch = args.eval_every_n_epoch
    config.lr = args.lr
    config.seed = args.seed
    config.gpu_ids = args.gpu_ids
    config.batch_size = args.batch_size
    config.batch_size_eval = args.batch_size_eval
    config.checkpoint_start = args.checkpoint_start
    config.train_mode = args.train_mode
    config.test_mode = args.test_mode
    config.label_smoothing = args.label_smoothing
    config.k = args.k
    config.train_ratio = args.train_ratio
    config.prob_flip = args.prob_flip
    config.use_pose_attention = args.use_pose_attention
    config.zero_shot = not args.no_zero_shot
    if args.rotate_query_to_north and args.no_rotate_query_to_north:
        raise ValueError("Cannot set both --rotate_query_to_north and --no_rotate_query_to_north.")
    config.rotate_query_to_north = args.rotate_query_to_north
    config.train_random_rotate90 = args.train_random_rotate90
    config.pose_attn_floor = args.pose_attn_floor
    config.pose_gate_mode = args.pose_gate_mode
    config.pose_gate_lambda = args.pose_gate_lambda
    config.pose_gate_insert_stage = args.pose_gate_insert_stage
    config.pose_fov_deg = args.pose_fov_deg
    config.custom_sampling = not args.no_custom_sampling
    config.use_wandb = not args.no_wandb

    train_script(config)
