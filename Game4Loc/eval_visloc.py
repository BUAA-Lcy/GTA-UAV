import argparse
import atexit
import logging
import os
from dataclasses import dataclass


def parse_tuple(s):
    try:
        return tuple(map(int, s.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Tuple must be integers separated by commas") from exc


def parse_rotate_step(value):
    try:
        step = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Rotate step must be a non-negative number") from exc
    if step < 0:
        raise argparse.ArgumentTypeError("Rotate step must be >= 0")
    return step


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value_str = str(value).strip().lower()
    if value_str in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value_str in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected, for example True or False")


def normalize_angle_deg(angle):
    return ((float(angle) + 180.0) % 360.0) - 180.0


def build_query_yaw_list(dataset, indices):
    raw_yaws = getattr(dataset, "images_yaw", None)
    if raw_yaws is None:
        return None

    query_yaws = []
    for index in indices:
        if index >= len(raw_yaws):
            query_yaws.append(None)
            continue
        yaw_value = raw_yaws[index]
        if yaw_value is None:
            query_yaws.append(None)
            continue
        try:
            yaw_value = float(yaw_value)
        except (TypeError, ValueError):
            query_yaws.append(None)
            continue
        # VisLoc default convention: satellite is north-up, so query drone images
        # should be rotated by -Phi1 (clockwise by Phi1 degrees when Phi1 > 0).
        query_yaws.append(normalize_angle_deg(-yaw_value))
    return query_yaws


@dataclass
class Configuration:
    # Model
    model: str = "vit_base_patch16_rope_reg1_gap_256.sbb_in1k"
    img_size: int = 384
    share_weights: bool = True

    # Evaluation
    batch_size: int = 128
    batch_size_eval: int = 128
    verbose: bool = True
    gpu_ids: tuple = (0,)
    normalize_features: bool = True
    eval_gallery_n: int = -1

    # With Fine Matching
    with_match: bool = False
    match_mode: str = "sparse"
    rotate: float = 0.0
    use_yaw: bool = False
    sparse_angle_score_inlier_offset: int = 25
    multi_scale: bool = True
    sparse_save_final_vis: bool = True
    angle_experiment: bool = False

    test_mode: str = "pos"
    query_mode: str = "D2S"

    # Dataset
    dataset: str = "VisLoc-D2S"
    checkpoint_start: str = "work_dir/visloc/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/1220150328/weights_end.pth"

    plot_acc_threshold: bool = False
    top10_log: bool = False
    query_limit: int = 0
    iteration: bool = False

    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == "nt" else 4

    # train on GPU if available
    device: str = ""

    data_root: str = "./data/UAV_VisLoc_dataset"
    test_pairs_meta_file: str = "cross-area-drone2sate-test.json"
    sate_img_dir: str = "satellite"
    use_wandb: bool = True


def eval_script(config):
    import torch
    from torch.utils.data import DataLoader, Subset

    from game4loc.dataset.visloc import VisLocDatasetEval, get_transforms
    from game4loc.evaluate.visloc import evaluate
    from game4loc.logger_utils import log_config, log_run_header, log_timer, setup_logger
    from game4loc.models.model import DesModel
    from game4loc.wandb_utils import WandbStepTimer, finish_wandb, init_wandb_run, safe_log

    if not config.device:
        config.device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_name = "VisLoc"
    area_tag = "cross" if "cross" in str(config.test_pairs_meta_file) else "same"
    match_tag = "match_on" if config.with_match else "match_off"
    run_type_str = f"eval_{dataset_name}_{area_tag}_{match_tag}"

    logger, log_path = setup_logger(
        algorithm_name=config.model,
        log_level=logging.DEBUG,
        logger_name="game4loc.eval.visloc",
        run_type=run_type_str,
        dataset_name=dataset_name,
    )
    log_run_header(logger, run_mode="test", algorithm_name=config.model)
    logger.info("自动日志路径: %s", log_path)
    logger.info("评估区域模式: %s-area", area_tag)
    logger.info("匹配模块: %s", "开启" if config.with_match else "关闭")
    if config.with_match:
        logger.info("with_match 子步骤模式: %s", config.match_mode)
        if config.rotate > 0:
            if config.match_mode == "sparse":
                logger.info("旋转搜索步长: 一阶段=%.1f° 二阶段=%.1f°", config.rotate, config.rotate / 2.0)
            else:
                logger.info("旋转搜索步长: %.1f°", config.rotate)
        else:
            logger.info("旋转搜索: 关闭")
        if config.match_mode == "sparse":
            logger.info("稀疏匹配偏航角 (yaw) 先验: %s", "启用 (如果数据提供)" if config.use_yaw else "关闭 (默认仅做旋转搜索)")
            if config.use_yaw:
                logger.info("VisLoc yaw 默认约定: 使用 -Phi1 对齐正北卫星图 (即 Phi1>0 时查询航拍图顺时针旋转 Phi1°)")
            logger.info("VisLoc 稀疏最佳角度评分: score = ratio * max(inliers - %d, 0)", config.sparse_angle_score_inlier_offset)
            logger.info("VisLoc 稀疏多尺度匹配: %s", "开启" if config.multi_scale else "关闭")
            logger.info("VisLoc 稀疏匹配使用内置稳定默认参数: phase2关闭, RANSAC=RANSAC, reproj=20, SP.det=0.003, SP.kpts=2048, SP.nms=4, scales=(1.0, 0.8, 0.6, 1.2)")
            logger.info(
                "VisLoc 稀疏最终匹配可视化: %s (目录=%s, 最多=%d张)",
                "开启" if config.sparse_save_final_vis else "关闭",
                "Log/visloc_sparse_final_matches",
                200,
            )
            logger.info("VisLoc 角度实验日志: %s", "开启" if config.angle_experiment else "关闭")
            if config.use_yaw and config.rotate <= 0:
                logger.info("当前 rotate<=0，将不执行任何旋转，因此 yaw 先验不会生效")
    log_config(logger, config)

    wandb_run = None
    if config.use_wandb:
        wandb_run = init_wandb_run(
            config=config,
            algorithm_name=f"{config.model}_eval_visloc",
            dataset_name=dataset_name,
            run_type="eval",
        )
        wandb_run.config.update({"run_type": "eval", "batch_size_eval": config.batch_size_eval}, allow_val_change=True)
        atexit.register(finish_wandb, wandb_run)

    logger.info("模型: %s", config.model)

    with log_timer(logger, "测试模型初始化", level=logging.INFO, sync_cuda=True), \
         WandbStepTimer("eval_visloc/model_initialization", run=wandb_run, sync_cuda=True):
        model = DesModel(
            config.model,
            pretrained=True,
            img_size=config.img_size,
            share_weights=config.share_weights,
        )

    data_config = model.get_config()
    logger.debug("模型数据配置: %s", data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = (config.img_size, config.img_size)

    if config.checkpoint_start is not None:
        logger.info("加载权重: %s", config.checkpoint_start)
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

    with log_timer(logger, "测试数据加载与构建", level=logging.INFO), \
         WandbStepTimer("eval_visloc/data_loading", run=wandb_run):
        val_transforms, _, _ = get_transforms(img_size, mean=mean, std=std)

        if config.query_mode == "D2S":
            query_dataset_test = VisLocDatasetEval(
                data_root=config.data_root,
                pairs_meta_file=config.test_pairs_meta_file,
                view="drone",
                mode=config.test_mode,
                transforms=val_transforms,
                query_mode=config.query_mode,
            )
            gallery_dataset_test = VisLocDatasetEval(
                data_root=config.data_root,
                pairs_meta_file=config.test_pairs_meta_file,
                view="sate",
                transforms=val_transforms,
                sate_img_dir=config.sate_img_dir,
                query_mode=config.query_mode,
            )
            pairs_dict = query_dataset_test.pairs_drone2sate_dict
        else:
            gallery_dataset_test = VisLocDatasetEval(
                data_root=config.data_root,
                pairs_meta_file=config.test_pairs_meta_file,
                view="drone",
                mode=config.test_mode,
                transforms=val_transforms,
                query_mode=config.query_mode,
            )
            pairs_dict = gallery_dataset_test.pairs_sate2drone_dict
            query_dataset_test = VisLocDatasetEval(
                data_root=config.data_root,
                pairs_meta_file=config.test_pairs_meta_file,
                view="sate",
                transforms=val_transforms,
                sate_img_dir=config.sate_img_dir,
                query_mode=config.query_mode,
                pairs_sate2drone_dict=pairs_dict,
            )

    if config.iteration:
        config.query_limit = len(query_dataset_test) // 10
        logger.info("启用 --iteration 模式，将仅评估前 %d 张查询图像 (约 1/10 的数据)", config.query_limit)

    query_yaw_list = None
    if config.query_limit is not None and config.query_limit > 0:
        query_indices = list(range(min(config.query_limit, len(query_dataset_test))))
        base_query_dataset = query_dataset_test
        query_dataset_test = Subset(query_dataset_test, query_indices)
        query_img_list = [base_query_dataset.images_name[i] for i in query_indices]
        query_center_loc_xy_list = [base_query_dataset.images_center_loc_xy[i] for i in query_indices]
        if config.query_mode == "D2S" and config.use_yaw:
            query_yaw_list = build_query_yaw_list(base_query_dataset, query_indices)
    else:
        query_img_list = query_dataset_test.images_name
        query_center_loc_xy_list = query_dataset_test.images_center_loc_xy
        if config.query_mode == "D2S" and config.use_yaw:
            query_yaw_list = build_query_yaw_list(query_dataset_test, range(len(query_dataset_test)))

    gallery_center_loc_xy_list = gallery_dataset_test.images_center_loc_xy
    gallery_topleft_loc_xy_list = gallery_dataset_test.images_topleft_loc_xy
    gallery_img_list = gallery_dataset_test.images_name

    query_dataloader_test = DataLoader(
        query_dataset_test,
        batch_size=config.batch_size_eval,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True,
    )
    gallery_dataloader_test = DataLoader(
        gallery_dataset_test,
        batch_size=config.batch_size_eval,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    logger.info("测试查询图像总数: %s", len(query_dataset_test))
    logger.info("测试图库图像总数: %s", len(gallery_dataset_test))

    dis_threshold_list = [10 * (i + 1) for i in range(50)] if "cross" in config.test_pairs_meta_file else [4 * (i + 1) for i in range(50)]

    effective_with_match = config.with_match
    if config.query_mode != "D2S" and config.with_match:
        logger.warning("VisLoc 的 with_match 仅支持 D2S 评估，当前 query_mode=%s，将自动关闭 with_match。", config.query_mode)
        effective_with_match = False
    effective_angle_experiment = config.angle_experiment
    if effective_angle_experiment and (not effective_with_match or config.match_mode != "sparse"):
        logger.warning("角度实验仅支持 VisLoc 的 with_match+sparse 评估，当前配置将自动关闭 angle_experiment。")
        effective_angle_experiment = False

    logger.info("%s[UAV-VisLoc 测试评估]%s", 30 * "-", 30 * "-")

    with log_timer(logger, "测试阶段总体评估", level=logging.INFO, sync_cuda=True), \
         WandbStepTimer("eval_visloc/evaluation", run=wandb_run, sync_cuda=True):
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
            pairs_dict=pairs_dict,
            query_yaw_list=query_yaw_list,
            ranks_list=[1, 5, 10],
            step_size=1000,
            dis_threshold_list=dis_threshold_list,
            cleanup=True,
            plot_acc_threshold=config.plot_acc_threshold,
            top10_log=config.top10_log,
            with_match=effective_with_match,
            match_mode=config.match_mode,
            rotate=config.rotate,
            sparse_angle_score_inlier_offset=config.sparse_angle_score_inlier_offset,
            sparse_use_multi_scale=config.multi_scale,
            sparse_save_final_vis=config.sparse_save_final_vis,
            angle_experiment=effective_angle_experiment,
            wandb_run=wandb_run,
            logger=logger,
        )

    safe_log(wandb_run, {"eval/recall@1": float(r1_test) * 100.0})
    logger.info("测试结束，Recall@1=%.4f", float(r1_test) * 100.0)
    finish_wandb(wandb_run)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation script for VisLoc.")

    parser.add_argument("--data_root", type=str, default="./data/UAV_VisLoc_dataset", help="Data root")
    parser.add_argument("--test_pairs_meta_file", type=str, default="cross-area-drone2sate-test.json", help="Test metafile path")
    parser.add_argument("--sate_img_dir", type=str, default="satellite", help="Satellite image directory")
    parser.add_argument("--model", type=str, default="vit_base_patch16_rope_reg1_gap_256.sbb_in1k", help="Model architecture")
    parser.add_argument("--checkpoint_start", type=str, default=None, help="Checkpoint path for evaluation")
    parser.add_argument("--gpu_ids", type=parse_tuple, default=(0,), help="GPU IDs, for example: 0 or 0,1")
    parser.add_argument("--batch_size", type=int, default=128, help="Evaluation batch size")
    parser.add_argument("--num_workers", type=int, default=None, help="Dataloader workers (default: platform-specific config value)")
    parser.add_argument("--test_mode", type=str, default="pos", help="Test with pair in iou or oc")
    parser.add_argument("--query_mode", type=str, default="D2S", choices=("D2S", "S2D"), help="Retrieval direction")
    parser.add_argument("--no_share_weights", action="store_true", help="Evaluate without shared backbone weights")
    parser.add_argument("--with_match", action="store_true", help="Test with post-process image matching")
    match_group = parser.add_mutually_exclusive_group()
    match_group.add_argument("--dense", dest="match_mode", action="store_const", const="dense", help="Use dense matching in with_match step")
    match_group.add_argument("--sparse", dest="match_mode", action="store_const", const="sparse", help="Use sparse matching in with_match step")
    parser.set_defaults(match_mode="sparse")
    parser.add_argument("--use_yaw", action="store_true", help="Use yaw information during sparse matching")
    parser.add_argument("--ignore_yaw", action="store_false", dest="use_yaw", help=argparse.SUPPRESS)
    parser.add_argument("--rotate", type=parse_rotate_step, nargs="?", const=90.0, default=0.0, help="Enable rotation search with an optional step in degrees. '--rotate' defaults to 90. In sparse mode phase 2 uses half of this value. Default is 0 when the flag is omitted, which disables all rotation.")
    parser.add_argument("--no_rotate", action="store_const", const=0.0, dest="rotate", help=argparse.SUPPRESS)
    parser.add_argument("--multi_scale", type=parse_bool, nargs="?", const=True, default=True, help="Enable sparse multi-scale matching. Default is True. Use '--multi_scale False' to disable.")
    parser.add_argument("--angle_experiment", action="store_true", help="Log detailed per-angle sparse matching results for each VisLoc sample")
    parser.add_argument("--query_limit", type=int, default=0, help="Limit the number of queries for quick evaluation (0 for all)")
    parser.add_argument("--iteration", action="store_true", help="Only evaluate on a fixed 1/10 query subset for faster iteration")
    parser.add_argument("--plot_acc_threshold", action="store_true", help="Print accuracy-threshold curve values after evaluation")
    parser.add_argument("--top10_log", action="store_true", help="Print top-10 retrieval details for each query")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = Configuration()
    config.data_root = args.data_root
    config.test_pairs_meta_file = args.test_pairs_meta_file
    config.sate_img_dir = args.sate_img_dir
    config.model = args.model
    config.checkpoint_start = args.checkpoint_start
    config.gpu_ids = args.gpu_ids
    config.batch_size = args.batch_size
    config.batch_size_eval = args.batch_size
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    config.test_mode = args.test_mode
    config.query_mode = args.query_mode
    config.share_weights = not args.no_share_weights
    config.with_match = args.with_match
    config.match_mode = args.match_mode
    config.use_yaw = args.use_yaw
    config.rotate = args.rotate
    config.multi_scale = args.multi_scale
    config.angle_experiment = args.angle_experiment
    config.query_limit = args.query_limit
    config.iteration = args.iteration
    config.plot_acc_threshold = args.plot_acc_threshold
    config.top10_log = args.top10_log
    config.dataset = f"VisLoc-{config.query_mode}"

    eval_script(config)
