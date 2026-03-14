import os
import atexit
import logging
import torch
import argparse
from dataclasses import dataclass
from torch.utils.data import DataLoader

from game4loc.dataset.gta import GTADatasetEval, get_transforms
from game4loc.evaluate.gta import evaluate
from game4loc.models.model import DesModel
from game4loc.logger_utils import setup_logger, log_config, log_timer, log_run_header
from game4loc.wandb_utils import init_wandb_run, finish_wandb, WandbStepTimer, safe_log


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
    
    # Override model image size
    img_size: int = 384
    
    # Evaluation
    batch_size: int = 128
    verbose: bool = True
    gpu_ids: tuple = (0)
    normalize_features: bool = True

    # With Fine Matching
    with_match: bool = False

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
    use_wandb: bool = True


def eval_script(config):
    wandb_run = None
    logger, log_path = setup_logger(algorithm_name=config.model, log_level=logging.DEBUG, logger_name="game4loc.eval")
    log_run_header(logger, run_mode="test", algorithm_name=config.model)
    logger.info("自动日志路径: %s", log_path)
    log_config(logger, config)
    if config.use_wandb:
        wandb_run = init_wandb_run(config=config, algorithm_name=f"{config.model}_eval", logger=logger)
        wandb_run.config.update({"run_type": "eval", "batch_size": config.batch_size}, allow_val_change=True)
        atexit.register(finish_wandb, wandb_run, logger)

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
    
    logger.info("模型: %s", config.model)

    with log_timer(logger, "测试模型初始化", level=logging.INFO, sync_cuda=True), \
         WandbStepTimer("eval_gta/model_initialization", logger=logger, run=wandb_run, sync_cuda=True):
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

    with log_timer(logger, "测试数据加载与构建", level=logging.INFO), \
         WandbStepTimer("eval_gta/data_loading", logger=logger, run=wandb_run):
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
    query_img_list = query_dataset_test.images_name
    query_center_loc_xy_list = query_dataset_test.images_center_loc_xy

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
    with log_timer(logger, "测试阶段总体评估", level=logging.INFO, sync_cuda=True), \
         WandbStepTimer("eval_gta/evaluation", logger=logger, run=wandb_run, sync_cuda=True):
        r1_test = evaluate(config=config,
                               model=model,
                               query_loader=query_dataloader_test,
                               gallery_loader=gallery_dataloader_test,
                               query_list=query_img_list,
                               gallery_list=gallery_img_list,
                               pairs_dict=pairs_dict,
                               ranks_list=[1, 5, 10],
                               query_center_loc_xy_list=query_center_loc_xy_list,
                               gallery_center_loc_xy_list=gallery_center_loc_xy_list,
                               gallery_topleft_loc_xy_list=gallery_topleft_loc_xy_list,
                               step_size=1000,
                               dis_threshold_list=dis_threshold_list,
                               cleanup=True,
                               plot_acc_threshold=False,
                               top10_log=False,
                               with_match=config.with_match,
                               logger=logger,
                               wandb_run=wandb_run)
    logger.info("测试结束，Recall@1=%.4f", r1_test * 100.0)
    safe_log(wandb_run, {"eval/recall@1": float(r1_test) * 100.0})
    finish_wandb(wandb_run, logger=logger)
 


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for gta.")

    parser.add_argument('--log_to_file', action='store_true', help='Log saving to file')

    parser.add_argument('--log_path', type=str, default=None, help='Log file path')

    parser.add_argument('--data_root', type=str, default='./data/GTA-UAV-data', help='Data root')
   
    parser.add_argument('--test_pairs_meta_file', type=str, default='cross-area-drone2sate-test.json', help='Test metafile path')

    parser.add_argument('--model', type=str, default='vit_base_patch16_rope_reg1_gap_256.sbb_in1k', help='Model architecture')

    parser.add_argument('--no_share_weights', action='store_true', help='Model not sharing wieghts')

    parser.add_argument('--with_match', action='store_true', help='Test with post-process image matching (GIM, etc)')

    parser.add_argument('--gpu_ids', type=parse_tuple, default=(0,1), help='GPU ID')

    parser.add_argument('--batch_size', type=int, default=40, help='Batch size')

    parser.add_argument('--checkpoint_start', type=str, default=None, help='Training from checkpoint')

    parser.add_argument('--test_mode', type=str, default='pos', help='Test with positive pairs')

    parser.add_argument('--query_mode', type=str, default='D2S', help='Retrieval with drone to satellite')
    parser.add_argument('--no_wandb', action='store_true', help='Disable Weights & Biases logging')

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
    config.gpu_ids = args.gpu_ids
    config.checkpoint_start = args.checkpoint_start
    config.model = args.model
    config.share_weights = not(args.no_share_weights)
    config.test_mode = args.test_mode
    config.query_mode = args.query_mode
    config.with_match = args.with_match
    config.use_wandb = not(args.no_wandb)

    eval_script(config)