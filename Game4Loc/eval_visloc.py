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


def parse_float_list(value):
    items = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            items.append(float(token))
        except ValueError as exc:
            raise argparse.ArgumentTypeError("Float list expected, for example 1.0,0.8,0.6") from exc
    if not items:
        raise argparse.ArgumentTypeError("At least one scale value is required")
    return tuple(items)


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
    sparse_angle_score_inlier_offset: float | None = None
    multi_scale: bool = True
    sparse_scales: tuple = (1.0, 0.8, 0.6, 1.2)
    sparse_multi_scale_mode: str = "both"
    sparse_allow_upsample: bool = False
    sparse_cross_scale_dedup_radius: float = 0.0
    sparse_lightglue_profile: str = "current"
    sparse_sp_detection_threshold: float = 0.0003
    sparse_sp_max_num_keypoints: int = 4096
    sparse_sp_nms_radius: int = 4
    sparse_ransac_method: str = "RANSAC"
    sparse_secondary_on_fallback: bool = False
    sparse_secondary_ransac_method: str = "RANSAC"
    sparse_secondary_mode: str = "per_candidate"
    sparse_secondary_accept_min_inliers: int = 0
    sparse_secondary_accept_min_inlier_ratio: float = 0.0
    sparse_ransac_reproj_threshold: float = 20.0
    sparse_min_inliers: int = 15
    sparse_min_inlier_ratio: float = 0.001
    sparse_save_final_vis: bool = True
    angle_experiment: bool = False
    orientation_checkpoint: str = ""
    orientation_mode: str = "off"
    orientation_fusion_weight: float = 0.5
    orientation_topk: int = 1
    confidence_checkpoint: str = ""
    confidence_threshold: float = 0.5
    confidence_dump_path: str = ""

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
            if config.sparse_angle_score_inlier_offset is None:
                logger.info("VisLoc 稀疏旋转候选筛选: 按 inlier count 选最优，同分时按 inlier ratio 打破平局")
            else:
                logger.info(
                    "VisLoc 稀疏旋转候选评分: score = ratio * max(inliers - %.1f, 0)",
                    float(config.sparse_angle_score_inlier_offset),
                )
            logger.info("VOP 方向后验模式: %s", config.orientation_mode)
            if config.orientation_mode != "off":
                logger.info("VOP 权重路径: %s", config.orientation_checkpoint)
                logger.info("VOP 融合权重: %.3f", config.orientation_fusion_weight)
                logger.info("VOP top-k 假设数: %d", int(config.orientation_topk))
            if config.orientation_mode == "prior_topk" and str(config.confidence_checkpoint).strip():
                logger.info("Confidence verifier 权重路径: %s", config.confidence_checkpoint)
                logger.info("Confidence verifier 阈值: %.3f", float(config.confidence_threshold))
            if str(config.confidence_dump_path).strip():
                logger.info("Confidence dump 输出路径: %s", config.confidence_dump_path)
            logger.info("VisLoc 稀疏多尺度匹配: %s", "开启" if config.multi_scale else "关闭")
            logger.info("VisLoc 稀疏尺度列表: %s", ",".join(f"{float(scale):.2f}" for scale in config.sparse_scales))
            logger.info("VisLoc 稀疏多尺度缩放模式: %s", config.sparse_multi_scale_mode)
            logger.info("VisLoc 稀疏允许上采样: %s", "开启" if config.sparse_allow_upsample else "关闭")
            logger.info("VisLoc 稀疏跨尺度去重半径: %.2f px", float(config.sparse_cross_scale_dedup_radius))
            logger.info("VisLoc 稀疏 LightGlue 配置档: %s", config.sparse_lightglue_profile)
            logger.info(
                "VisLoc 稀疏匹配参数: phase2关闭, RANSAC=%s, reproj=%.3f, SP.det=%.4g, SP.kpts=%d, SP.nms=%d, min_inliers=%d, min_inlier_ratio=%.6f",
                config.sparse_ransac_method,
                float(config.sparse_ransac_reproj_threshold),
                float(config.sparse_sp_detection_threshold),
                int(config.sparse_sp_max_num_keypoints),
                int(config.sparse_sp_nms_radius),
                int(config.sparse_min_inliers),
                float(config.sparse_min_inlier_ratio),
            )
            if config.sparse_secondary_on_fallback:
                logger.info(
                    "VisLoc 稀疏双路径回退: 开启 (primary=%s -> secondary=%s, mode=%s, 仅当 primary 回退到 coarse center 时触发)",
                    config.sparse_ransac_method,
                    config.sparse_secondary_ransac_method,
                    config.sparse_secondary_mode,
                )
            logger.info(
                "VisLoc 稀疏最终匹配可视化: %s (目录=%s, 最多=%d张)",
                "开启" if config.sparse_save_final_vis else "关闭",
                "Log/visloc_sparse_final_matches",
                200,
            )
            logger.info("VisLoc 角度实验日志: %s", "开启" if config.angle_experiment else "关闭")
            if config.use_yaw and config.rotate <= 0:
                logger.info("当前 rotate<=0，将执行 yaw-only 对齐，不再额外做旋转搜索")
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
            sparse_scales=config.sparse_scales,
            sparse_multi_scale_mode=config.sparse_multi_scale_mode,
            sparse_allow_upsample=config.sparse_allow_upsample,
            sparse_cross_scale_dedup_radius=config.sparse_cross_scale_dedup_radius,
            sparse_lightglue_profile=config.sparse_lightglue_profile,
            sparse_sp_detection_threshold=config.sparse_sp_detection_threshold,
            sparse_sp_max_num_keypoints=config.sparse_sp_max_num_keypoints,
            sparse_sp_nms_radius=config.sparse_sp_nms_radius,
            sparse_ransac_method=config.sparse_ransac_method,
            sparse_secondary_on_fallback=config.sparse_secondary_on_fallback,
            sparse_secondary_ransac_method=config.sparse_secondary_ransac_method,
            sparse_secondary_mode=config.sparse_secondary_mode,
            sparse_secondary_accept_min_inliers=config.sparse_secondary_accept_min_inliers,
            sparse_secondary_accept_min_inlier_ratio=config.sparse_secondary_accept_min_inlier_ratio,
            sparse_ransac_reproj_threshold=config.sparse_ransac_reproj_threshold,
            sparse_min_inliers=config.sparse_min_inliers,
            sparse_min_inlier_ratio=config.sparse_min_inlier_ratio,
            sparse_save_final_vis=config.sparse_save_final_vis,
            angle_experiment=effective_angle_experiment,
            orientation_checkpoint=config.orientation_checkpoint,
            orientation_mode=config.orientation_mode,
            orientation_fusion_weight=config.orientation_fusion_weight,
            orientation_topk=config.orientation_topk,
            confidence_checkpoint=config.confidence_checkpoint,
            confidence_threshold=config.confidence_threshold,
            confidence_dump_path=config.confidence_dump_path,
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
    match_group.add_argument("--loftr", dest="match_mode", action="store_const", const="loftr", help="Use LoFTR matching in with_match step")
    parser.set_defaults(match_mode="sparse")
    parser.add_argument("--use_yaw", action="store_true", help="Use yaw information during sparse matching")
    parser.add_argument("--ignore_yaw", action="store_false", dest="use_yaw", help=argparse.SUPPRESS)
    parser.add_argument("--rotate", type=parse_rotate_step, nargs="?", const=90.0, default=0.0, help="Enable rotation search with an optional step in degrees. '--rotate' defaults to 90. In sparse mode phase 2 uses half of this value. Default is 0 when the flag is omitted, which disables all rotation.")
    parser.add_argument("--no_rotate", action="store_const", const=0.0, dest="rotate", help=argparse.SUPPRESS)
    parser.add_argument(
        "--sparse_angle_score_inlier_offset",
        type=float,
        default=None,
        help="Optional sparse rotate-candidate score offset. Leave unset to select by inlier count only; set a value to use ratio * max(inliers-offset, 0).",
    )
    parser.add_argument("--multi_scale", type=parse_bool, nargs="?", const=True, default=True, help="Enable sparse multi-scale matching. Default is True. Use '--multi_scale False' to disable.")
    parser.add_argument("--sparse_scales", type=parse_float_list, default=(1.0, 0.8, 0.6, 1.2), help="Comma-separated scale multipliers for sparse matching, for example 1.0,0.8,0.6,1.2")
    parser.add_argument("--sparse_multi_scale_mode", type=str, default="both", choices=("both", "query_only", "gallery_only"), help="How sparse multi-scale resizing is applied. Default is both.")
    parser.add_argument("--sparse_allow_upsample", type=parse_bool, nargs="?", const=True, default=False, help="Allow sparse scales above 1.0 to enlarge the image up to the matcher max edge. Default is False.")
    parser.add_argument("--sparse_cross_scale_dedup_radius", type=float, default=0.0, help="Greedy dedup radius in pixels after concatenating sparse matches across scales. Default is 0, which disables dedup.")
    parser.add_argument("--sparse_lightglue_profile", type=str, default="current", choices=("current", "official_default", "minima_ref"), help="Sparse LightGlue config profile. 'official_default' follows the current upstream LightGlue defaults; 'minima_ref' matches the MINIMA-style LightGlue settings.")
    parser.add_argument("--sparse_sp_detection_threshold", type=float, default=0.0003, help="Sparse SuperPoint detection threshold.")
    parser.add_argument("--sparse_sp_max_num_keypoints", type=int, default=4096, help="Sparse SuperPoint max_num_keypoints.")
    parser.add_argument("--sparse_sp_nms_radius", type=int, default=4, help="Sparse SuperPoint NMS radius.")
    parser.add_argument("--sparse_ransac_method", type=str, default="RANSAC", choices=("RANSAC", "USAC_FAST", "USAC_MAGSAC", "USAC_PROSAC", "USAC_DEFAULT", "USAC_FM_8PTS", "USAC_ACCURATE", "USAC_PARALLEL"), help="Sparse homography estimator method.")
    parser.add_argument("--sparse_secondary_on_fallback", type=parse_bool, nargs="?", const=True, default=False, help="Retry sparse fine localization with a secondary matcher only when the primary path falls back to coarse center.")
    parser.add_argument("--sparse_secondary_ransac_method", type=str, default="RANSAC", choices=("RANSAC", "USAC_FAST", "USAC_MAGSAC", "USAC_PROSAC", "USAC_DEFAULT", "USAC_FM_8PTS", "USAC_ACCURATE", "USAC_PARALLEL"), help="Sparse secondary homography estimator method used by the optional fallback retry.")
    parser.add_argument("--sparse_secondary_mode", type=str, default="per_candidate", choices=("per_candidate", "final_only"), help="How the optional sparse secondary fallback is applied.")
    parser.add_argument("--sparse_secondary_accept_min_inliers", type=int, default=0, help="Additional minimum inliers required before accepting a secondary sparse retry result.")
    parser.add_argument("--sparse_secondary_accept_min_inlier_ratio", type=float, default=0.0, help="Additional minimum inlier ratio required before accepting a secondary sparse retry result.")
    parser.add_argument("--sparse_ransac_reproj_threshold", type=float, default=20.0, help="Sparse homography RANSAC reprojection threshold.")
    parser.add_argument("--sparse_min_inliers", type=int, default=15, help="Minimum sparse inliers required to keep the estimated homography.")
    parser.add_argument("--sparse_min_inlier_ratio", type=float, default=0.001, help="Minimum sparse inlier ratio required to keep the estimated homography.")
    parser.add_argument("--sparse_save_final_vis", type=parse_bool, nargs="?", const=True, default=True, help="Save final sparse match visualizations. Default is True. Use '--sparse_save_final_vis False' to disable.")
    parser.add_argument("--angle_experiment", action="store_true", help="Log detailed per-angle sparse matching results for each VisLoc sample")
    parser.add_argument("--orientation_checkpoint", type=str, default="", help="Checkpoint path for the visual orientation posterior head")
    parser.add_argument("--orientation_mode", type=str, default="off", choices=("off", "single", "fusion", "prior_single", "prior_topk"), help="How to use the visual orientation posterior")
    parser.add_argument("--orientation_fusion_weight", type=float, default=0.5, help="Weight of the VOP posterior when orientation_mode=fusion")
    parser.add_argument("--orientation_topk", type=int, default=1, help="Number of VOP angle hypotheses to evaluate when orientation_mode=prior_topk")
    parser.add_argument("--confidence_checkpoint", type=str, default="", help="Checkpoint path for the lightweight confidence-aware verifier used after prior_topk")
    parser.add_argument("--confidence_threshold", type=float, default=0.5, help="Acceptance threshold for the confidence-aware verifier")
    parser.add_argument("--confidence_dump_path", type=str, default="", help="Optional JSON path for dumping top-k candidate verification records")
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
    config.sparse_angle_score_inlier_offset = None if args.sparse_angle_score_inlier_offset is None else float(args.sparse_angle_score_inlier_offset)
    config.multi_scale = args.multi_scale
    config.sparse_scales = tuple(float(scale) for scale in args.sparse_scales)
    config.sparse_multi_scale_mode = args.sparse_multi_scale_mode
    config.sparse_allow_upsample = bool(args.sparse_allow_upsample)
    config.sparse_cross_scale_dedup_radius = float(args.sparse_cross_scale_dedup_radius)
    config.sparse_lightglue_profile = args.sparse_lightglue_profile
    config.sparse_sp_detection_threshold = float(args.sparse_sp_detection_threshold)
    config.sparse_sp_max_num_keypoints = int(args.sparse_sp_max_num_keypoints)
    config.sparse_sp_nms_radius = int(args.sparse_sp_nms_radius)
    config.sparse_ransac_method = str(args.sparse_ransac_method)
    config.sparse_secondary_on_fallback = bool(args.sparse_secondary_on_fallback)
    config.sparse_secondary_ransac_method = str(args.sparse_secondary_ransac_method)
    config.sparse_secondary_mode = str(args.sparse_secondary_mode)
    config.sparse_secondary_accept_min_inliers = int(args.sparse_secondary_accept_min_inliers)
    config.sparse_secondary_accept_min_inlier_ratio = float(args.sparse_secondary_accept_min_inlier_ratio)
    config.sparse_ransac_reproj_threshold = float(args.sparse_ransac_reproj_threshold)
    config.sparse_min_inliers = int(args.sparse_min_inliers)
    config.sparse_min_inlier_ratio = float(args.sparse_min_inlier_ratio)
    config.sparse_save_final_vis = bool(args.sparse_save_final_vis)
    config.angle_experiment = args.angle_experiment
    config.orientation_checkpoint = args.orientation_checkpoint
    config.orientation_mode = args.orientation_mode
    config.orientation_fusion_weight = args.orientation_fusion_weight
    config.orientation_topk = max(1, int(args.orientation_topk))
    config.confidence_checkpoint = args.confidence_checkpoint
    config.confidence_threshold = float(args.confidence_threshold)
    config.confidence_dump_path = args.confidence_dump_path
    config.query_limit = args.query_limit
    config.iteration = args.iteration
    config.plot_acc_threshold = args.plot_acc_threshold
    config.top10_log = args.top10_log
    config.dataset = f"VisLoc-{config.query_mode}"

    eval_script(config)
