import os
import atexit
import logging
import argparse
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader

from game4loc.dataset.visloc import VisLocDatasetEval, get_transforms
from game4loc.evaluate.visloc import evaluate
from game4loc.models.model import DesModel
from game4loc.wandb_utils import init_wandb_run, finish_wandb, WandbStepTimer, safe_log
from game4loc.logger_utils import setup_logger, log_config, log_run_header, log_timer


def parse_tuple(s):
    try:
        return tuple(map(int, s.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Tuple must be integers separated by commas") from exc


@dataclass
class Configuration:
    model: str = "vit_base_patch16_rope_reg1_gap_256.sbb_in1k"
    img_size: int = 384
    batch_size_eval: int = 128
    verbose: bool = True
    gpu_ids: tuple = (0,)
    normalize_features: bool = True
    eval_gallery_n: int = -1
    with_match: bool = False
    test_mode: str = "pos"
    checkpoint_start: str = None
    num_workers: int = 0 if os.name == "nt" else 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    use_pose_attention: bool = False
    rotate_query_to_north: bool = False
    pose_attn_floor: float = 0.3
    pose_gate_mode: str = "multiplicative"
    pose_gate_lambda: float = 0.5
    pose_gate_insert_stage: str = "pre_blocks"
    pose_fov_deg: float = 36.0

    data_root: str = "./data/UAV_VisLoc_dataset"
    test_pairs_meta_file: str = "same-area-drone2sate-test-pose.json"
    sate_img_dir: str = "satellite"
    use_wandb: bool = True


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation script for UAV-VisLoc with optional pose-driven prior.")
    parser.add_argument("--data_root", type=str, default="./data/UAV_VisLoc_dataset")
    parser.add_argument("--test_pairs_meta_file", type=str, default="same-area-drone2sate-test-pose.json")
    parser.add_argument("--checkpoint_start", type=str, required=True)
    parser.add_argument("--model", type=str, default="vit_base_patch16_rope_reg1_gap_256.sbb_in1k")
    parser.add_argument("--batch_size_eval", type=int, default=128)
    parser.add_argument("--gpu_ids", type=parse_tuple, default=(0,))
    parser.add_argument("--test_mode", type=str, default="pos")
    parser.add_argument("--with_match", action="store_true")
    parser.add_argument("--use_pose_attention", action="store_true")
    parser.add_argument("--rotate_query_to_north", action="store_true")
    parser.add_argument("--no_rotate_query_to_north", action="store_true")
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
    parser.add_argument("--no_wandb", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = Configuration()
    config.data_root = args.data_root
    config.test_pairs_meta_file = args.test_pairs_meta_file
    config.checkpoint_start = args.checkpoint_start
    config.model = args.model
    config.batch_size_eval = args.batch_size_eval
    config.gpu_ids = args.gpu_ids
    config.test_mode = args.test_mode
    config.with_match = args.with_match
    config.use_pose_attention = args.use_pose_attention
    if args.rotate_query_to_north and args.no_rotate_query_to_north:
        raise ValueError("Cannot set both --rotate_query_to_north and --no_rotate_query_to_north.")
    config.rotate_query_to_north = args.rotate_query_to_north
    config.pose_attn_floor = args.pose_attn_floor
    config.pose_gate_mode = args.pose_gate_mode
    config.pose_gate_lambda = args.pose_gate_lambda
    config.pose_gate_insert_stage = args.pose_gate_insert_stage
    config.pose_fov_deg = args.pose_fov_deg
    config.use_wandb = not args.no_wandb

    dataset_name = "UAV-VisLoc"
    logger, log_path = setup_logger(
        algorithm_name=config.model,
        log_level=logging.DEBUG,
        logger_name="game4loc.eval.visloc_pose",
        run_type="eval",
        dataset_name=dataset_name,
    )
    log_run_header(logger, run_mode="test", algorithm_name=config.model)
    logger.info("自动日志路径: %s", log_path)
    log_config(logger, config)

    wandb_run = None
    if config.use_wandb:
        wandb_run = init_wandb_run(
            config=config,
            algorithm_name=f"{config.model}_eval_visloc",
            dataset_name=dataset_name,
            run_type="eval",
        )
        wandb_run.config.update(
            {
                "run_type": "eval",
                "batch_size_eval": config.batch_size_eval,
            },
            allow_val_change=True,
        )
        atexit.register(finish_wandb, wandb_run)

    logger.info("模型: %s", config.model)
    logger.info("Use pose attention: %s", config.use_pose_attention)
    logger.info("Rotate UAV query to north by Phi1: %s", config.rotate_query_to_north)

    with log_timer(logger, "eval_model_initialization", level=logging.INFO, sync_cuda=True), \
         WandbStepTimer("eval_visloc_pose/model_initialization", run=wandb_run, sync_cuda=True):
        model = DesModel(
            config.model,
            pretrained=True,
            img_size=config.img_size,
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

    logger.info("加载权重: %s", config.checkpoint_start)
    model_state_dict = torch.load(config.checkpoint_start)
    model.load_state_dict(model_state_dict, strict=False)

    logger.info("可用 GPU 数量: %s", torch.cuda.device_count())
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)

    model = model.to(config.device)

    with log_timer(logger, "eval_data_loading", level=logging.INFO), \
         WandbStepTimer("eval_visloc_pose/data_loading", run=wandb_run):
        val_transforms, _, _ = get_transforms(img_size, mean=mean, std=std, random_rotate90=False)

        query_dataset_test = VisLocDatasetEval(
            data_root=config.data_root,
            pairs_meta_file=config.test_pairs_meta_file,
            view="drone",
            mode=config.test_mode,
            transforms=val_transforms,
            use_pose=config.use_pose_attention,
            rotate_aerial_to_north=config.rotate_query_to_north,
        )
        pairs_drone2sate_dict = query_dataset_test.pairs_drone2sate_dict
        query_img_list = query_dataset_test.images_name
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
        gallery_center_loc_xy_list = gallery_dataset_test.images_center_loc_xy
        gallery_topleft_loc_xy_list = gallery_dataset_test.images_topleft_loc_xy
        gallery_img_list = gallery_dataset_test.images_name
        gallery_dataloader_test = DataLoader(
            gallery_dataset_test,
            batch_size=config.batch_size_eval,
            num_workers=config.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    logger.info("测试查询图像总数: %s", len(query_dataset_test))
    logger.info("测试图库图像总数: %s", len(gallery_dataset_test))

    if "cross" in config.test_pairs_meta_file:
        dis_threshold_list = [10 * (i + 1) for i in range(50)]
    else:
        dis_threshold_list = [4 * (i + 1) for i in range(50)]

    logger.info("%s[UAV-VisLoc 测试评估]%s", 30 * "-", 30 * "-")
    with log_timer(logger, "eval_visloc_pose/evaluation", level=logging.INFO, sync_cuda=True), \
         WandbStepTimer("eval_visloc_pose/evaluation", run=wandb_run, sync_cuda=True):
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
            dis_threshold_list=dis_threshold_list,
            cleanup=True,
            plot_acc_threshold=True,
            top10_log=True,
            with_match=config.with_match,
            wandb_run=wandb_run,
            logger=logger,
        )

    safe_log(wandb_run, {"eval/recall@1": float(r1_test) * 100.0})
    logger.info("测试结束，Recall@1=%.4f", float(r1_test) * 100.0)
    finish_wandb(wandb_run)
