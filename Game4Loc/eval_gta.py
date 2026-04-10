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
        dataset_name=""
    )
    log_run_header(logger, run_mode="test", algorithm_name=config.model)
    logger.info("自动日志路径: %s", log_path)
    logger.info("评估区域模式: %s-area", area_tag)
    logger.info("匹配模块: %s", "开启" if config.with_match else "关闭")
    if config.with_match:
        logger.info("with_match 子步骤模式: %s", config.match_mode)
        if config.match_mode == 'sparse':
            logger.info("稀疏匹配偏航角 (yaw) 先验: %s", "启用 (如果数据提供)" if config.use_yaw else "关闭 (默认仅做旋转搜索)")
            logger.info("稀疏匹配四向旋转搜索 (rotate): %s", "启用" if config.rotate else "关闭")
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

    if config.query_limit is not None and config.query_limit > 0:
        # 为了保证每次跑的数据相同，这里固定取前 query_limit 张图像
        query_indices = list(range(min(config.query_limit, len(query_dataset_test))))
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
    parser.set_defaults(match_mode='sparse')
    parser.add_argument('--orientation_checkpoint', type=str, default='', help='Checkpoint path for the visual orientation posterior head')
    parser.add_argument('--orientation_mode', type=str, default='off', choices=('off', 'prior_single', 'prior_topk'), help='How to use the visual orientation posterior in GTA sparse fine localization')
    parser.add_argument('--orientation_fusion_weight', type=float, default=0.5, help='Reserved for API parity; not used by prior_* modes')
    parser.add_argument('--orientation_topk', type=int, default=1, help='Number of VOP angle hypotheses to evaluate when orientation_mode=prior_topk')
    parser.add_argument('--save_match_vis', action='store_true', help='Save sparse final match visualizations for each query')
    parser.add_argument('--match_vis_dir', type=str, default='', help='Directory for sparse final match visualizations')
    parser.add_argument('--match_vis_max_save', type=int, default=200, help='Maximum number of sparse final match visualizations to save')
    parser.add_argument('--query_limit', type=int, default=0, help='Limit the number of queries for quick evaluation (0 for all)')

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
    parser.add_argument('--no_rotate', action='store_true', help='Disable 4-way rotation search in sparse mode. By default rotation search is enabled.')

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
    config.query_limit = args.query_limit
    config.use_wandb = False # 强制关闭wandb
    config.use_yaw = args.use_yaw
    config.iteration = args.iteration
    config.rotate = not args.no_rotate

    eval_script(config)
