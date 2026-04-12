import os
import atexit
import logging
import torch
import argparse
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from game4loc.dataset.gta import GTADatasetEval, get_transforms
from game4loc.evaluate.gta import evaluate
from game4loc.models.model import DesModel
from game4loc.logger_utils import setup_logger, log_config, log_timer, log_run_header


def parse_tuple(s):
    try:
        return tuple(map(int, s.split(',')))
    except ValueError:
        raise argparse.ArgumentTypeError("Tuple must be integers separated by commas")


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value_str = str(value).strip().lower()
    if value_str in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value_str in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")


def parse_float_list(value):
    if isinstance(value, (list, tuple)):
        return tuple(float(v) for v in value)
    raw = str(value).strip()
    if not raw:
        return tuple()
    try:
        return tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Float list expected, got: {value}") from exc


@dataclass
class Configuration:

    # Model
    # model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    model: str = 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k'
    share_weights: bool = True
    
    # Override model image size
    img_size: int = 384
    
    # Evaluation
    batch_size: int = 128
    verbose: bool = True
    gpu_ids: tuple = (0)
    normalize_features: bool = True

    # With Fine Matching
    with_match: bool = False
    match_mode: str = "sparse"
    orientation_checkpoint: str = ""
    orientation_mode: str = "off"
    orientation_fusion_weight: float = 0.5
    orientation_topk: int = 1
    save_match_vis: bool = False
    match_vis_dir: str = ""
    match_vis_max_save: int = 200
    sparse_angle_score_inlier_offset: float | None = None
    sparse_use_multi_scale: bool = True
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

    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4 
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 

    # Dataset
    query_mode: str = 'D2S'
    # query_mode: str = 'S2D'

    # Checkpoint to start from
    # checkpoint_start = '/home/xmuairmud/jyx/GTA-UAV/Game4Loc/pretrained/gta/same_area/selavpr.pth'
    checkpoint_start = 'pretrained/gta/cross_area/game4loc.pth'

    # data_root: str = "/home/xmuairmud/data/GTA-UAV-data/GTA-UAV-Lidar/GTA-UAV-Lidar"
    data_root: str = "/home/xmuairmud/data/GTA-UAV-data/GTA-UAV-official/GTA-UAV-LR-hf"

    train_pairs_meta_file = 'cross-area-drone2sate-train.json'
    test_pairs_meta_file = 'cross-area-drone2sate-test.json'
    sate_img_dir = 'satellite'
    use_wandb: bool = False
    use_yaw: bool = False
    iteration: bool = False
    rotate: bool = True
    query_limit: int = 0
    query_offset: int = 0


def eval_script(config):
    dataset_name = "GTA-UAV"
    area_tag = "cross" if ('cross' in str(config.test_pairs_meta_file)) else "same"
    match_tag = "match_on" if config.with_match else "match_off"
    
    # 构建基础名称，例如 eval_GTA-UAV_cross_match_on
    run_type_str = f"eval_{dataset_name}_{area_tag}_{match_tag}"

    logger, log_path = setup_logger(
        algorithm_name=config.model, 
        log_level=logging.DEBUG, 
        logger_name="game4loc.eval", 
        run_type=run_type_str, 
        dataset_name="",
        log_file_path=config.log_path,
    )
    log_run_header(logger, run_mode="test", algorithm_name=config.model)
    logger.info("自动日志路径: %s", log_path)
    logger.info("评估区域模式: %s-area", area_tag)
    logger.info("匹配模块: %s", "开启" if config.with_match else "关闭")
    if config.with_match:
        logger.info("with_match 子步骤模式: %s", config.match_mode)
        if config.match_mode == 'sparse':
            logger.info("稀疏匹配偏航角 (yaw) 先验: %s", "启用 (如果数据提供)" if config.use_yaw else "关闭 (默认仅做旋转搜索)")
            logger.info("稀疏多尺度匹配: %s", "开启" if config.sparse_use_multi_scale else "关闭")
            logger.info("稀疏尺度列表: %s", ",".join(f"{float(scale):.3f}" for scale in config.sparse_scales))
            logger.info("稀疏多尺度缩放模式: %s", config.sparse_multi_scale_mode)
            logger.info("稀疏允许上采样: %s", "开启" if config.sparse_allow_upsample else "关闭")
            logger.info("稀疏跨尺度去重半径: %.2f px", float(config.sparse_cross_scale_dedup_radius))
            logger.info("稀疏 LightGlue 配置档: %s", config.sparse_lightglue_profile)
            logger.info("稀疏 SuperPoint detection_threshold: %.6f", float(config.sparse_sp_detection_threshold))
            logger.info("稀疏 SuperPoint max_num_keypoints: %d", int(config.sparse_sp_max_num_keypoints))
            logger.info("稀疏 SuperPoint nms_radius: %d", int(config.sparse_sp_nms_radius))
            logger.info("稀疏 Homography RANSAC method: %s", str(config.sparse_ransac_method))
            logger.info("稀疏 Homography RANSAC threshold: %.3f", float(config.sparse_ransac_reproj_threshold))
            logger.info("稀疏 H 质量门槛: min_inliers=%d min_inlier_ratio=%.6f", int(config.sparse_min_inliers), float(config.sparse_min_inlier_ratio))
            if bool(config.sparse_secondary_on_fallback):
                logger.info(
                    "GTA 稀疏双路径回退: 开启 (primary=%s -> secondary=%s, mode=%s, accept_min_inliers=%d, accept_min_inlier_ratio=%.4f)",
                    str(config.sparse_ransac_method),
                    str(config.sparse_secondary_ransac_method),
                    str(config.sparse_secondary_mode),
                    int(config.sparse_secondary_accept_min_inliers),
                    float(config.sparse_secondary_accept_min_inlier_ratio),
                )
            logger.info("稀疏匹配四向旋转搜索 (rotate): %s", "启用" if config.rotate else "关闭")
            logger.info("稀疏旋转候选筛选: 按 inlier count 选最优，同分时按 inlier ratio 打破平局")
            logger.info("VOP 方向后验模式: %s", config.orientation_mode)
            if config.orientation_mode != "off":
                logger.info("VOP 权重路径: %s", config.orientation_checkpoint)
                logger.info("VOP 融合权重: %.3f", config.orientation_fusion_weight)
                logger.info("VOP top-k 假设数: %d", int(config.orientation_topk))
            logger.info("匹配可视化导出: %s", "开启" if config.save_match_vis else "关闭")
            if config.save_match_vis:
                logger.info("匹配可视化目录: %s", config.match_vis_dir if str(config.match_vis_dir).strip() else "(默认 Log/visloc_sparse_final_matches)")
                logger.info("匹配可视化最大保存数: %d", int(config.match_vis_max_save))
        elif config.orientation_mode != "off":
            logger.warning("当前仅 sparse 匹配路径支持 VOP；match_mode=%s 时会自动忽略 orientation_mode=%s", config.match_mode, config.orientation_mode)
    log_config(logger, config)

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
    
    logger.info("模型: %s", config.model)

    with log_timer(logger, "测试模型初始化", level=logging.INFO, sync_cuda=True):
        model = DesModel(config.model,
                        pretrained=False,
                        img_size=config.img_size,
                        share_weights=config.share_weights)
                          
    data_config = model.get_config()
    logger.debug("模型数据配置: %s", data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = (config.img_size, config.img_size)
    

    # load pretrained Checkpoint    
    if config.checkpoint_start is not None:  
        with log_timer(logger, "加载测试检查点", level=logging.INFO, sync_cuda=True):
            logger.info("加载权重文件: %s", config.checkpoint_start)
            model_state_dict = torch.load(config.checkpoint_start)
            model.load_state_dict(model_state_dict, strict=True)

    # Data parallel
    logger.info("可用 GPU 数量: %s", torch.cuda.device_count())
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
            
    # Model to device   
    model = model.to(config.device)

    logger.info("查询图像尺寸: %s", img_size)
    logger.info("图库图像尺寸: %s", img_size)
    logger.info("归一化均值 Mean: %s", mean)
    logger.info("归一化方差 Std: %s", std)


    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#

    with log_timer(logger, "测试数据加载与构建", level=logging.INFO):
        val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(img_size, mean=mean, std=std)

        # Test query
        if config.query_mode == 'D2S':
            query_dataset_test = GTADatasetEval(data_root=config.data_root,
                                                pairs_meta_file=config.test_pairs_meta_file,
                                                view="drone",
                                                transforms=val_transforms,
                                                mode='pos',
                                                query_mode=config.query_mode,
                                                )
            gallery_dataset_test = GTADatasetEval(data_root=config.data_root,
                                                pairs_meta_file=config.test_pairs_meta_file,
                                                view="sate",
                                                transforms=val_transforms,
                                                sate_img_dir=config.sate_img_dir,
                                                mode='pos',
                                                query_mode=config.query_mode,
                                                )
            pairs_dict = query_dataset_test.pairs_drone2sate_dict
        elif config.query_mode == 'S2D':
            gallery_dataset_test = GTADatasetEval(data_root=config.data_root,
                                                pairs_meta_file=config.test_pairs_meta_file,
                                                view="drone",
                                                transforms=val_transforms,
                                                mode='pos',
                                                query_mode=config.query_mode,
                                                )
            pairs_dict = gallery_dataset_test.pairs_sate2drone_dict
            query_dataset_test = GTADatasetEval(data_root=config.data_root,
                                                pairs_meta_file=config.test_pairs_meta_file,
                                                view="sate",
                                                transforms=val_transforms,
                                                query_mode=config.query_mode,
                                                pairs_sate2drone_dict=pairs_dict,
                                                sate_img_dir=config.sate_img_dir,
                                                mode='pos',
                                            )
    # Optional limit on number of queries for quick evaluation
    if config.iteration:
        config.query_limit = len(query_dataset_test) // 10
        logger.info("启用 --iteration 模式，将仅评估前 %d 张查询图像 (约 1/10 的数据)", config.query_limit)

    total_query_count = len(query_dataset_test)
    query_offset = max(int(getattr(config, "query_offset", 0)), 0)
    if query_offset > 0:
        logger.info("启用查询分片偏移: query_offset=%d", query_offset)

    if (config.query_limit is not None and config.query_limit > 0) or query_offset > 0:
        query_start = min(query_offset, total_query_count)
        if config.query_limit is not None and config.query_limit > 0:
            query_end = min(query_start + int(config.query_limit), total_query_count)
        else:
            query_end = total_query_count
        query_indices = list(range(query_start, query_end))
        logger.info(
            "本次评估查询切片: start=%d end=%d count=%d / total=%d",
            query_start,
            query_end,
            len(query_indices),
            total_query_count,
        )
        query_dataset_test = Subset(query_dataset_test, query_indices)
        # derive lists from subset
        full_q_names = getattr(query_dataset_test.dataset, "images_name", [])
        full_q_locs = getattr(query_dataset_test.dataset, "images_center_loc_xy", [])
        full_q_yaws = getattr(query_dataset_test.dataset, "images_yaw", [])
        query_img_list = [full_q_names[i] for i in query_indices]
        query_center_loc_xy_list = [full_q_locs[i] for i in query_indices]
        query_yaw_list = [full_q_yaws[i] for i in query_indices] if (len(full_q_yaws) > 0 and config.use_yaw) else None
    else:
        query_img_list = query_dataset_test.images_name
        query_center_loc_xy_list = query_dataset_test.images_center_loc_xy
        query_yaw_list = getattr(query_dataset_test, "images_yaw", None) if config.use_yaw else None

    gallery_center_loc_xy_list = gallery_dataset_test.images_center_loc_xy
    gallery_topleft_loc_xy_list = gallery_dataset_test.images_topleft_loc_xy
    gallery_img_list = gallery_dataset_test.images_name

    query_dataloader_test = DataLoader(query_dataset_test,
                                    batch_size=config.batch_size,
                                    num_workers=config.num_workers,
                                    shuffle=False,
                                    pin_memory=True)
    gallery_dataloader_test = DataLoader(gallery_dataset_test,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    logger.info("测试查询图像总数: %s", len(query_dataset_test))
    logger.info("测试图库图像总数: %s", len(gallery_dataset_test))
    logger.debug("测试配对字典条目数: %s", len(pairs_dict))

    # For Test Log (distance threshold) 
    dis_threshold_list = None
    if 'cross' in config.test_pairs_meta_file:
        ####### Cross-area for total 500m/10m
        logger.info("评估模式: 跨区域评估（cross-area）")
        dis_threshold_list = [10*(i+1) for i in range(50)]
    else:
        ####### Same-area for total 200m/4m
        logger.info("评估模式: 同区域评估（same-area）")
        dis_threshold_list = [4*(i+1) for i in range(50)]
    
    logger.info("%s[开始 GTA-UAV 测试评估]%s", 30*"-", 30*"-")
    with log_timer(logger, "测试阶段总体评估", level=logging.INFO, sync_cuda=True):
        r1_test = evaluate(
            config=config,
            model=model,
            query_loader=query_dataloader_test,
            gallery_loader=gallery_dataloader_test,
            query_list=query_img_list,
            gallery_list=gallery_img_list,
            pairs_dict=pairs_dict,
            ranks_list=[1, 5, 10],
            query_center_loc_xy_list=query_center_loc_xy_list,
            query_yaw_list=query_yaw_list,
            gallery_center_loc_xy_list=gallery_center_loc_xy_list,
            gallery_topleft_loc_xy_list=gallery_topleft_loc_xy_list,
            step_size=1000,
            dis_threshold_list=dis_threshold_list,
            cleanup=True,
            plot_acc_threshold=False,
            top10_log=False,
            with_match=config.with_match,
            match_mode=config.match_mode,
            orientation_checkpoint=config.orientation_checkpoint,
            orientation_mode=config.orientation_mode,
            orientation_fusion_weight=config.orientation_fusion_weight,
            orientation_topk=config.orientation_topk,
            save_match_vis=config.save_match_vis,
            match_vis_dir=config.match_vis_dir,
            match_vis_max_save=config.match_vis_max_save,
            sparse_angle_score_inlier_offset=config.sparse_angle_score_inlier_offset,
            sparse_use_multi_scale=config.sparse_use_multi_scale,
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
            logger=logger,
            rotate=config.rotate,
        )
    logger.info("测试结束，Recall@1=%.4f", r1_test * 100.0)
 


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for gta.")

    parser.add_argument('--log_to_file', action='store_true', help='Log saving to file')

    parser.add_argument('--log_path', type=str, default=None, help='Log file path')

    parser.add_argument('--data_root', type=str, default='./data/GTA-UAV-data', help='Data root')
   
    parser.add_argument('--test_pairs_meta_file', type=str, default='cross-area-drone2sate-test.json', help='Test metafile path')

    parser.add_argument('--model', type=str, default='vit_base_patch16_rope_reg1_gap_256.sbb_in1k', help='Model architecture')

    parser.add_argument('--no_share_weights', action='store_true', help='Model not sharing wieghts')

    parser.add_argument('--with_match', action='store_true', help='Test with post-process image matching (GIM, etc)')
    match_group = parser.add_mutually_exclusive_group()
    match_group.add_argument('--dense', dest='match_mode', action='store_const', const='dense', help='Use dense matching in with_match step (original behavior)')
    match_group.add_argument('--sparse', dest='match_mode', action='store_const', const='sparse', help='Use sparse fast path in with_match step')
    match_group.add_argument('--loftr', dest='match_mode', action='store_const', const='loftr', help='Use LoFTR matching in with_match step')
    parser.set_defaults(match_mode='sparse')
    parser.add_argument('--orientation_checkpoint', type=str, default='', help='Checkpoint path for the visual orientation posterior head')
    parser.add_argument('--orientation_mode', type=str, default='off', choices=('off', 'prior_single', 'prior_topk'), help='How to use the visual orientation posterior in GTA sparse fine localization')
    parser.add_argument('--orientation_fusion_weight', type=float, default=0.5, help='Reserved for API parity; not used by prior_* modes')
    parser.add_argument('--orientation_topk', type=int, default=1, help='Number of VOP angle hypotheses to evaluate when orientation_mode=prior_topk')
    parser.add_argument('--save_match_vis', action='store_true', help='Save sparse final match visualizations for each query')
    parser.add_argument('--match_vis_dir', type=str, default='', help='Directory for sparse final match visualizations')
    parser.add_argument('--match_vis_max_save', type=int, default=200, help='Maximum number of sparse final match visualizations to save')
    parser.add_argument('--sparse_angle_score_inlier_offset', type=float, default=None, help='Optional sparse rotate-candidate score offset. Leave unset to select by inlier count only.')
    parser.add_argument('--sparse_use_multi_scale', type=parse_bool, nargs='?', const=True, default=True, help='Enable sparse multi-scale matching. Default is True.')
    parser.add_argument('--sparse_scales', type=parse_float_list, default=(1.0, 0.8, 0.6, 1.2), help='Comma-separated sparse scale multipliers, for example 1.0,0.8,0.6,1.2')
    parser.add_argument('--sparse_multi_scale_mode', type=str, default='both', choices=('both', 'query_only', 'gallery_only'), help='How sparse multi-scale resizing is applied. Default is both.')
    parser.add_argument('--sparse_allow_upsample', type=parse_bool, nargs='?', const=True, default=False, help='Allow sparse scales above 1.0 to enlarge the image up to the matcher max edge. Default is False.')
    parser.add_argument('--sparse_cross_scale_dedup_radius', type=float, default=0.0, help='Greedy dedup radius in pixels after concatenating sparse matches across scales. Default is 0.')
    parser.add_argument('--sparse_lightglue_profile', type=str, default='current', choices=('current', 'official_default', 'minima_ref'), help='Sparse LightGlue config profile.')
    parser.add_argument('--sparse_sp_detection_threshold', type=float, default=0.0003, help='Sparse SuperPoint detection threshold.')
    parser.add_argument('--sparse_sp_max_num_keypoints', type=int, default=4096, help='Sparse SuperPoint max_num_keypoints.')
    parser.add_argument('--sparse_sp_nms_radius', type=int, default=4, help='Sparse SuperPoint NMS radius.')
    parser.add_argument('--sparse_ransac_method', type=str, default='RANSAC', choices=('RANSAC', 'USAC_FAST', 'USAC_MAGSAC', 'USAC_PROSAC', 'USAC_DEFAULT', 'USAC_FM_8PTS', 'USAC_ACCURATE', 'USAC_PARALLEL'), help='Sparse homography estimator method.')
    parser.add_argument('--sparse_secondary_on_fallback', type=parse_bool, nargs='?', const=True, default=False, help='Retry sparse fine localization with a secondary matcher only when the primary path falls back to coarse center.')
    parser.add_argument('--sparse_secondary_ransac_method', type=str, default='RANSAC', choices=('RANSAC', 'USAC_FAST', 'USAC_MAGSAC', 'USAC_PROSAC', 'USAC_DEFAULT', 'USAC_FM_8PTS', 'USAC_ACCURATE', 'USAC_PARALLEL'), help='Sparse secondary homography estimator method used by the optional fallback retry.')
    parser.add_argument('--sparse_secondary_mode', type=str, default='per_candidate', choices=('per_candidate', 'final_only'), help='How the optional sparse secondary fallback is applied.')
    parser.add_argument('--sparse_secondary_accept_min_inliers', type=int, default=0, help='Additional minimum inliers required before accepting a secondary sparse retry result.')
    parser.add_argument('--sparse_secondary_accept_min_inlier_ratio', type=float, default=0.0, help='Additional minimum inlier ratio required before accepting a secondary sparse retry result.')
    parser.add_argument('--sparse_ransac_reproj_threshold', type=float, default=20.0, help='Sparse homography RANSAC reprojection threshold in pixels.')
    parser.add_argument('--sparse_min_inliers', type=int, default=15, help='Minimum inlier count required before accepting sparse homography.')
    parser.add_argument('--sparse_min_inlier_ratio', type=float, default=0.001, help='Minimum inlier ratio required before accepting sparse homography.')
    parser.add_argument('--query_limit', type=int, default=0, help='Limit the number of queries for quick evaluation (0 for all)')
    parser.add_argument('--query_offset', type=int, default=0, help='Skip the first N queries before applying query_limit. Default is 0.')

    parser.add_argument('--gpu_ids', type=parse_tuple, default=(0,1), help='GPU ID')

    parser.add_argument('--batch_size', type=int, default=40, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of dataloader workers (default: use config default)')

    parser.add_argument('--checkpoint_start', type=str, default=None, help='Training from checkpoint')

    parser.add_argument('--test_mode', type=str, default='pos', help='Test with positive pairs')

    parser.add_argument('--query_mode', type=str, default='D2S', help='Retrieval with drone to satellite')
    parser.add_argument('--no_wandb', action='store_true', help='Disable Weights & Biases logging')
    parser.add_argument('--use_yaw', action='store_true', help='Use yaw information during sparse matching')
    parser.add_argument('--ignore_yaw', action='store_false', dest='use_yaw', help=argparse.SUPPRESS)
    parser.add_argument('--iteration', action='store_true', help='If True, only evaluate on 1/10 of the query data (fixed subset) for faster iteration.')
    parser.add_argument('--no_rotate', action='store_true', help='Disable 4-way rotation search in fine localization. By default GTA sparse/dense matching uses rotate90 search when this flag is omitted.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    config = Configuration()
    config.data_root = args.data_root
    config.test_pairs_meta_file = args.test_pairs_meta_file
    config.log_to_file = args.log_to_file
    config.log_path = args.log_path
    config.batch_size = args.batch_size
    if args.num_workers is not None:
        config.num_workers = int(args.num_workers)
    config.gpu_ids = args.gpu_ids
    config.checkpoint_start = args.checkpoint_start
    config.model = args.model
    config.share_weights = not(args.no_share_weights)
    config.test_mode = args.test_mode
    config.query_mode = args.query_mode
    config.with_match = args.with_match
    config.match_mode = args.match_mode
    config.orientation_checkpoint = args.orientation_checkpoint
    config.orientation_mode = args.orientation_mode
    config.orientation_fusion_weight = args.orientation_fusion_weight
    config.orientation_topk = max(1, int(args.orientation_topk))
    config.save_match_vis = args.save_match_vis
    config.match_vis_dir = args.match_vis_dir
    config.match_vis_max_save = max(1, int(args.match_vis_max_save))
    config.sparse_angle_score_inlier_offset = None if args.sparse_angle_score_inlier_offset is None else float(args.sparse_angle_score_inlier_offset)
    config.sparse_use_multi_scale = bool(args.sparse_use_multi_scale)
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
    config.query_limit = args.query_limit
    config.query_offset = args.query_offset
    config.use_wandb = False # 强制关闭wandb
    config.use_yaw = args.use_yaw
    config.iteration = args.iteration
    config.rotate = not args.no_rotate

    eval_script(config)
