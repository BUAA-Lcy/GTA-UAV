import os
import atexit
import logging
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader

from game4loc.dataset.visloc import VisLocDatasetEval, get_transforms
from game4loc.evaluate.visloc import evaluate
from game4loc.models.model import DesModel
from game4loc.wandb_utils import init_wandb_run, finish_wandb, WandbStepTimer, safe_log
from game4loc.logger_utils import setup_logger, log_config, log_run_header, log_timer


@dataclass
class Configuration:

    # Model
    # model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    model: str = 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k'
    # model: str = 'TransGeo'
    
    # Override model image size
    img_size: int = 384
    
    # Evaluation
    batch_size: int = 128
    batch_size_eval: int = 128
    verbose: bool = True
    gpu_ids: tuple = (0)
    normalize_features: bool = True
    eval_gallery_n: int = -1             # -1 for all or int

    # With Fine Matching
    with_match: bool = False

    test_mode: str = 'pos'

    query_mode = 'D2S'

    # Dataset
    dataset: str = 'VisLoc-D2S'           # 'U1652-D2S' | 'U1652-S2D'

    # Checkpoint to start from
    # checkpoint_start = 'work_dir/university/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/162436/weights_end.pth'
    # checkpoint_start = 'work_dir/sues/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0723160833/weights_end.pth'
    # checkpoint_start = 'work_dir/denseuav/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0723164458/weights_end.pth'
    # checkpoint_start = 'work_dir/gta/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0722110449/weights_end.pth'
    
    # checkpoint_start = 'work_dir/visloc/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0723224548/weights_end.pth' ## GTA-UAV
    # checkpoint_start = 'work_dir/visloc/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0724003322/weights_end.pth' ## DenseUAV
    # checkpoint_start = 'work_dir/visloc/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0724022105/weights_end.pth' ## SUES-200
    # checkpoint_start = 'work_dir/visloc/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0723205823/weights_end.pth' ## ImageNet
    # checkpoint_start = 'work_dir/visloc/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0724145818/weights_end.pth' ## University
    checkpoint_start = 'work_dir/visloc/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/1220150328/weights_end.pth'

    # checkpoint_start = './pretrained/visloc/same_area/transgeo.pth'

    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4 
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 

    data_root = '/home/xmuairmud/data/UAV_VisLoc_dataset_Lidar'

    train_pairs_meta_file = 'cross-area-drone2sate-train-z31.json'
    test_pairs_meta_file = 'cross-area-drone2sate-test-z31.json'
    sate_img_dir = 'satellite'

    dis_threshold_list = None
    if 'cross' in test_pairs_meta_file:
        ####### Cross-area
        print("cross-area eval")
        dis_threshold_list = [10*(i+1) for i in range(50)]
    else:
        ####### Same-area
        print("same-area eval")
        dis_threshold_list = [4*(i+1) for i in range(50)]
    use_wandb: bool = True


#-----------------------------------------------------------------------------#
# Config                                                                      #
#-----------------------------------------------------------------------------#

config = Configuration() 



if __name__ == '__main__':
    logger, log_path = setup_logger(algorithm_name=config.model, log_level=logging.DEBUG, logger_name="game4loc.eval.visloc", run_type="eval", dataset_name="VisLoc")
    log_run_header(logger, run_mode="test", algorithm_name=config.model)
    logger.info("自动日志路径: %s", log_path)
    log_config(logger, config)
    wandb_run = None
    if config.use_wandb:
        wandb_run = init_wandb_run(config=config, algorithm_name=f"{config.model}_eval_visloc", dataset_name="VisLoc", run_type="eval")
        wandb_run.config.update({"run_type": "eval", "batch_size_eval": config.batch_size_eval}, allow_val_change=True)
        atexit.register(finish_wandb, wandb_run)

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
    
    logger.info("模型: %s", config.model)


    with log_timer(logger, "测试模型初始化", level=logging.INFO, sync_cuda=True), \
         WandbStepTimer("eval_visloc/model_initialization", run=wandb_run, sync_cuda=True):
        model = DesModel(config.model,
                              pretrained=True,
                              img_size=config.img_size)
                          
    data_config = model.get_config()
    logger.debug("模型数据配置: %s", data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = (config.img_size, config.img_size)
    

    # load pretrained Checkpoint    
    if config.checkpoint_start is not None:  
        logger.info("加载权重: %s", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)  
        model.load_state_dict(model_state_dict, strict=False)     

    # Data parallel
    logger.info("可用 GPU 数量: %s", torch.cuda.device_count())
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
            
    # Model to device   
    model = model.to(config.device)

    logger.info("查询图像尺寸: %s", img_size)
    logger.info("图库图像尺寸: %s", img_size)
    logger.info("Mean: %s", mean)
    logger.info("Std: %s", std)


    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#

    # Transforms
    with log_timer(logger, "测试数据加载与构建", level=logging.INFO), \
         WandbStepTimer("eval_visloc/data_loading", run=wandb_run):
        val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(img_size, mean=mean, std=std)


    # Test query
    query_dataset_test = VisLocDatasetEval(data_root=config.data_root,
                                        pairs_meta_file=config.test_pairs_meta_file,
                                        view="drone",
                                        mode=config.test_mode,
                                        transforms=val_transforms,
                                        )
    pairs_drone2sate_dict = query_dataset_test.pairs_drone2sate_dict
    query_img_list = query_dataset_test.images_name
    query_center_loc_xy_list = query_dataset_test.images_center_loc_xy
    
    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)

    # Test gallery
    gallery_dataset_test = VisLocDatasetEval(data_root=config.data_root,
                                               pairs_meta_file=config.test_pairs_meta_file,
                                               view="sate",
                                               transforms=val_transforms,
                                               sate_img_dir=config.sate_img_dir,
                                               )
    gallery_center_loc_xy_list = gallery_dataset_test.images_center_loc_xy
    gallery_topleft_loc_xy_list = gallery_dataset_test.images_topleft_loc_xy
    gallery_img_list = gallery_dataset_test.images_name

    gallery_dataloader_test = DataLoader(gallery_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    logger.info("测试查询图像总数: %s", len(query_dataset_test))
    logger.info("测试图库图像总数: %s", len(gallery_dataset_test))
    
    logger.info("%s[UAV-VisLoc 测试评估]%s", 30*"-", 30*"-")

    with log_timer(logger, "测试阶段总体评估", level=logging.INFO, sync_cuda=True), \
         WandbStepTimer("eval_visloc/evaluation", run=wandb_run, sync_cuda=True):
        r1_test = evaluate(config=config,
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
                               dis_threshold_list=[4*(i+1) for i in range(50)],
                               cleanup=True,
                               plot_acc_threshold=True,
                               top10_log=True,
                               with_match=config.with_match,
                               wandb_run=wandb_run,
                               logger=logger)
    safe_log(wandb_run, {"eval/recall@1": float(r1_test) * 100.0})
    logger.info("测试结束，Recall@1=%.4f", float(r1_test) * 100.0)
    finish_wandb(wandb_run)
 
