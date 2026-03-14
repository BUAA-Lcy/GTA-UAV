import os
import atexit
import logging
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader

from game4loc.dataset.university import U1652DatasetEval, get_transforms
from game4loc.evaluate.university import evaluate
from game4loc.models.model import DesModel
from game4loc.wandb_utils import init_wandb_run, finish_wandb, WandbStepTimer, safe_log
from game4loc.logger_utils import setup_logger, log_config, log_run_header, log_timer


@dataclass
class Configuration:

    # Model
    model: str = 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k'
    
    # Override model image size
    img_size: int = 384
    
    # Evaluation
    batch_size: int = 128
    verbose: bool = True
    gpu_ids: tuple = (0,1)
    normalize_features: bool = True
    eval_gallery_n: int = -1             # -1 for all or int
    
    # Dataset
    dataset: str = 'U1652-D2S'           # 'U1652-D2S' | 'U1652-S2D'
    # data_folder: str = "./data/U1652"
    
    # Checkpoint to start from
    # checkpoint_start = 'pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth'
    checkpoint_start = 'work_dir/denseuav/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0809045532/weights_end.pth'
    # checkpoint_start = None
  
    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4 
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 
    use_wandb: bool = True
    

#-----------------------------------------------------------------------------#
# Config                                                                      #
#-----------------------------------------------------------------------------#

config = Configuration() 

if config.dataset == 'U1652-D2S':
    config.query_folder_train = '/home/xmuairmud/data/University-Release/train/satellite'
    config.gallery_folder_train = '/home/xmuairmud/data/University-Release/train/drone'   
    config.query_folder_test = '/home/xmuairmud/data/University-Release/test/query_drone' 
    config.gallery_folder_test = '/home/xmuairmud/data/University-Release/test/gallery_satellite'    
elif config.dataset == 'U1652-S2D':
    config.query_folder_train = './data/U1652/train/satellite'
    config.gallery_folder_train = './data/U1652/train/drone'    
    config.query_folder_test = './data/U1652/test/query_satellite'
    config.gallery_folder_test = './data/U1652/test/gallery_drone'


if __name__ == '__main__':
    logger, log_path = setup_logger(algorithm_name=config.model, log_level=logging.DEBUG, logger_name="game4loc.eval.university")
    log_run_header(logger, run_mode="test", algorithm_name=config.model)
    logger.info("自动日志路径: %s", log_path)
    log_config(logger, config)
    wandb_run = None
    if config.use_wandb:
        wandb_run = init_wandb_run(config=config, algorithm_name=f"{config.model}_eval_university")
        wandb_run.config.update({"run_type": "eval", "batch_size": config.batch_size}, allow_val_change=True)
        atexit.register(finish_wandb, wandb_run)

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
        
    logger.info("模型: %s", config.model)


    with log_timer(logger, "测试模型初始化", level=logging.INFO, sync_cuda=True), \
         WandbStepTimer("eval_university/model_initialization", run=wandb_run, sync_cuda=True):
        model = TimmModel(config.model,
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
         WandbStepTimer("eval_university/data_loading", run=wandb_run):
        val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(img_size, mean=mean, std=std)
                                                                                                                                 
    
    # Reference Satellite Images
    query_dataset_test = U1652DatasetEval(data_folder=config.query_folder_test,
                                               mode="query",
                                               transforms=val_transforms,
                                               )
    
    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    # Query Ground Images Test
    gallery_dataset_test = U1652DatasetEval(data_folder=config.gallery_folder_test,
                                               mode="gallery",
                                               transforms=val_transforms,
                                               sample_ids=query_dataset_test.get_sample_ids(),
                                               gallery_n=config.eval_gallery_n,
                                               )
    
    gallery_dataloader_test = DataLoader(gallery_dataset_test,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    logger.info("测试查询图像总数: %s", len(query_dataset_test))
    logger.info("测试图库图像总数: %s", len(gallery_dataset_test))
   

    logger.info("%s[University-1652 测试评估]%s", 30*"-", 30*"-")

    with log_timer(logger, "测试阶段总体评估", level=logging.INFO, sync_cuda=True), \
         WandbStepTimer("eval_university/evaluation", run=wandb_run, sync_cuda=True):
        r1_test = evaluate(config=config,
                           model=model,
                           query_loader=query_dataloader_test,
                           gallery_loader=gallery_dataloader_test, 
                           ranks=[1, 5, 10],
                           step_size=1000,
                           cleanup=True,
                           wandb_run=wandb_run,
                           logger=logger)
    safe_log(wandb_run, {"eval/recall@1": float(r1_test) * 100.0})
    logger.info("测试结束，Recall@1=%.4f", float(r1_test) * 100.0)
    finish_wandb(wandb_run)
 
