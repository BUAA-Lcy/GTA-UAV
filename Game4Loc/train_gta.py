import os
import time
import math
import shutil
import logging
import atexit
import torch
import argparse
from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup

from game4loc.dataset.gta import GTADatasetEval, GTADatasetTrain, get_transforms
from game4loc.utils import setup_system
from game4loc.logger_utils import setup_logger, log_config, log_timer
from game4loc.wandb_utils import init_wandb_run, finish_wandb, WandbStepTimer, safe_log
from game4loc.trainer.trainer import train, train_with_weight
from game4loc.evaluate.gta import evaluate
from game4loc.loss import InfoNCE, WeightedInfoNCE, GroupInfoNCE, TripletLoss
from game4loc.models.model import DesModel
from game4loc.models.model_netvlad import DesModelWithVLAD


def parse_tuple(s):
    try:
        return tuple(map(int, s.split(',')))
    except ValueError:
        raise argparse.ArgumentTypeError("Tuple must be integers separated by commas")


@dataclass
class Configuration:
    
    # Model
    model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    model_hub: str = 'timm'
    with_netvlad: bool = False
    
    # Override model image size
    img_size: int = 384
 
    # Please Ignore
    freeze_layers: bool = False
    frozen_stages = [0,0,0,0]

    # Training with sharing weights
    share_weights: bool = True
    
    # Training with weighted-InfoNCE
    with_weight: bool = True

    # Please Ignore
    train_in_group: bool = True
    group_len = 2
    # Please Ignore
    loss_type = ["whole_slice", "part_slice"]

    # Please Ignore
    train_with_mix_data: bool = False
    # Please Ignore
    train_with_recon: bool = False
    recon_weight: float = 0.1
    
    # Training 
    mixed_precision: bool = True
    custom_sampling: bool = True         # use custom sampling instead of random
    seed = 1
    epochs: int = 10
    batch_size: int = 40                # keep in mind real_batch_size = 2 * batch_size
    verbose: bool = False
    gpu_ids: tuple = (0,1)           # GPU ids for training

    # Training with sparse data
    train_ratio: float = 1.0

    # Eval
    batch_size_eval: int = 128
    eval_every_n_epoch: int = 1          # eval every n Epoch
    normalize_features: bool = True
    eval_gallery_n: int = -1             # -1 for all or int

    # Optimizer 
    clip_grad = 100.                     # None | float
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False     # Gradient Checkpointing
    
    # Loss
    label_smoothing: float = 0.1
    k: float = 3
    
    # Learning Rate
    lr: float = 0.001                    # 1 * 10^-4 for ViT | 1 * 10^-1 for CNN
    scheduler: str = "cosine"            # "polynomial" | "cosine" | "constant" | None
    warmup_epochs: int = 0.1
    lr_end: float = 0.0001               #  only for "polynomial"

    # Augment Images
    prob_flip: float = 0.5               # flipping the sat image and drone image simultaneously
    
    # Savepath for model checkpoints
    model_path: str = "./work_dir/gta"
    use_wandb: bool = True

    query_mode: str = "D2S"               # Retrieval in Drone to Satellite

    train_mode: str = "pos_semipos"       # Train with positive + semi-positive pairs
    test_mode: str = "pos"                # Test with positive pairs

    # Eval before training
    zero_shot: bool = True
    
    # Checkpoint to start from
    checkpoint_start = None

    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4 
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 
    
    # for better performance
    cudnn_benchmark: bool = True
    
    # make cudnn deterministic
    cudnn_deterministic: bool = False

    data_root: str = "./data/GTA-UAV-data"

    train_pairs_meta_file = 'cross-area-drone2sate-train.json'
    test_pairs_meta_file = 'cross-area-drone2sate-test.json'
    sate_img_dir = 'satellite'


#-----------------------------------------------------------------------------#
# Train Config                                                                #
#-----------------------------------------------------------------------------#

def train_script(config):
    logger, log_path = setup_logger(algorithm_name=config.model, log_level=logging.DEBUG)
    wandb_run = None

    save_time = "{}".format(time.strftime("%m%d%H%M%S"))
    model_path = "{}/{}/{}".format(config.model_path,
                                       config.model,
                                       save_time)

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copyfile(os.path.basename(__file__), "{}/train.py".format(model_path))

    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)
    logger.info("训练输出目录: %s", model_path)
    logger.info("自动日志路径: %s", log_path)
    logger.info("训练起始检查点: %s", config.checkpoint_start)
    log_config(logger, config)
    if config.use_wandb:
        wandb_run = init_wandb_run(config=config, algorithm_name=config.model, logger=logger)
        # 演示核心超参数写入 wandb.config
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

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
    logger.info("模型: %s", config.model)

    with log_timer(logger, "model_initialization", level=logging.INFO, sync_cuda=True), \
         WandbStepTimer("model_initialization", logger=logger, run=wandb_run, sync_cuda=True):
        if config.with_netvlad:
            model = DesModelWithVLAD(model_name=config.model,
                        pretrained=True,
                        img_size=config.img_size,
                        share_weights=config.share_weights)
        else:
            model = DesModel(model_name=config.model,
                            pretrained=True,
                            img_size=config.img_size,
                            share_weights=config.share_weights)
                        
    data_config = model.get_config()
    logger.debug("Model data config: %s", data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = (config.img_size, config.img_size)
    
    # Activate gradient checkpointing
    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)
    
    # Load pretrained Checkpoint    
    if config.checkpoint_start is not None:  
        with log_timer(logger, "load_checkpoint", level=logging.INFO, sync_cuda=True):
            logger.info("加载检查点: %s", config.checkpoint_start)
            model_state_dict = torch.load(config.checkpoint_start)
            model.load_state_dict(model_state_dict, strict=False)

    logger.info("是否冻结模型层: %s, 冻结阶段: %s", config.freeze_layers, config.frozen_stages)
    if config.freeze_layers:
        model.freeze_layers(config.frozen_stages)

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

    logger.info("是否使用自定义采样: %s", config.custom_sampling)


    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#

    with log_timer(logger, "data_loading", level=logging.INFO), \
         WandbStepTimer("data_loading", logger=logger, run=wandb_run):
        if 'cross-area' in config.train_pairs_meta_file:
            sat_rot = True
        else:
            sat_rot = False
        val_transforms, train_sat_transforms, train_drone_transforms = \
            get_transforms(img_size, mean=mean, std=std, sat_rot=sat_rot)

        train_dataset = GTADatasetTrain(data_root=config.data_root,
                                        pairs_meta_file=config.train_pairs_meta_file,
                                        transforms_query=train_drone_transforms,
                                        transforms_gallery=train_sat_transforms,
                                        group_len=config.group_len,
                                        prob_flip=config.prob_flip,
                                        shuffle_batch_size=config.batch_size,
                                        mode=config.train_mode,
                                        train_ratio=config.train_ratio,
                                        )

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=config.batch_size,
                                      num_workers=config.num_workers,
                                      shuffle=not config.custom_sampling,
                                      pin_memory=True)

        if config.query_mode == 'D2S':
            query_view = 'drone'
            gallery_view = 'sate'
        else:
            query_view = 'sate'
            gallery_view = 'drone'
        query_dataset_test = GTADatasetEval(data_root=config.data_root,
                                            pairs_meta_file=config.test_pairs_meta_file,
                                            view=query_view,
                                            transforms=val_transforms,
                                            mode=config.test_mode,
                                            sate_img_dir=config.sate_img_dir,
                                            query_mode=config.query_mode,
                                            )
        query_img_list = query_dataset_test.images_name
        query_center_loc_xy_list = query_dataset_test.images_center_loc_xy
        pairs_drone2sate_dict = query_dataset_test.pairs_drone2sate_dict

        query_dataloader_test = DataLoader(query_dataset_test,
                                           batch_size=config.batch_size_eval,
                                           num_workers=config.num_workers,
                                           shuffle=False,
                                           pin_memory=True)

        gallery_dataset_test = GTADatasetEval(data_root=config.data_root,
                                              pairs_meta_file=config.test_pairs_meta_file,
                                              view=gallery_view,
                                              transforms=val_transforms,
                                              mode=config.test_mode,
                                              sate_img_dir=config.sate_img_dir,
                                              query_mode=config.query_mode,
                                             )
        gallery_center_loc_xy_list = gallery_dataset_test.images_center_loc_xy
        gallery_topleft_loc_xy_list = gallery_dataset_test.images_topleft_loc_xy
        gallery_img_list = gallery_dataset_test.images_name

        gallery_dataloader_test = DataLoader(gallery_dataset_test,
                                           batch_size=config.batch_size_eval,
                                           num_workers=config.num_workers,
                                           shuffle=False,
                                           pin_memory=True)
    
    logger.info("测试查询集图像数: %s", len(query_dataset_test))
    logger.info("测试图库图像数: %s", len(gallery_dataset_test))
    
    #-----------------------------------------------------------------------------#
    # Loss                                                                        #
    #-----------------------------------------------------------------------------#
    logger.info("是否启用加权训练: %s, k=%s", config.with_weight, config.k)
    
    loss_function_normal = WeightedInfoNCE(
        device=config.device,
        label_smoothing=config.label_smoothing,
        k=config.k,
    )
    ## For TripletLoss
    # loss_function_normal = TripletLoss(device=config.device)

    if config.mixed_precision:
        scaler = GradScaler(init_scale=2.**10)
    else:
        scaler = None
        
    #-----------------------------------------------------------------------------#
    # optimizer                                                                   #
    #-----------------------------------------------------------------------------#

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


    #-----------------------------------------------------------------------------#
    # Scheduler                                                                   #
    #-----------------------------------------------------------------------------#

    train_steps = len(train_dataloader) * config.epochs
    warmup_steps = len(train_dataloader) * config.warmup_epochs
       
    if config.scheduler == "polynomial":
        logger.info("学习率调度器: polynomial - 最大LR: %s - 结束LR: %s", config.lr, config.lr_end)
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                              num_training_steps=train_steps,
                                                              lr_end = config.lr_end,
                                                              power=1.5,
                                                              num_warmup_steps=warmup_steps)
        
    elif config.scheduler == "cosine":
        logger.info("学习率调度器: cosine - 最大LR: %s", config.lr)
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_training_steps=train_steps,
                                                    num_warmup_steps=warmup_steps)
        
    elif config.scheduler == "constant":
        logger.info("学习率调度器: constant - 最大LR: %s", config.lr)
        scheduler =  get_constant_schedule_with_warmup(optimizer,
                                                       num_warmup_steps=warmup_steps)
           
    else:
        scheduler = None
        
    logger.info("Warmup 轮数: %s - Warmup 步数: %s", str(config.warmup_epochs).ljust(2), warmup_steps)
    logger.info("训练总轮数: %s - 训练总步数: %s", config.epochs, train_steps)
        
        
    #-----------------------------------------------------------------------------#
    # Zero Shot                                                                   #
    #-----------------------------------------------------------------------------#
    if config.zero_shot:
        logger.info("%s[Zero Shot]%s", 30*"-", 30*"-")
        with log_timer(logger, "zero_shot_evaluation", level=logging.INFO, sync_cuda=True), \
             WandbStepTimer("zero_shot_evaluation", logger=logger, run=wandb_run, sync_cuda=True):
            r1_test = evaluate(config=config,
                               model=model,
                               query_loader=query_dataloader_test,
                               gallery_loader=gallery_dataloader_test,
                               query_list=query_img_list,
                               gallery_list=gallery_img_list,
                               pairs_dict=pairs_drone2sate_dict,
                               ranks_list=[1, 5, 10],
                               query_center_loc_xy_list=query_center_loc_xy_list,
                               gallery_center_loc_xy_list=gallery_center_loc_xy_list,
                               gallery_topleft_loc_xy_list=gallery_topleft_loc_xy_list,
                               step_size=1000,
                               cleanup=True,
                               logger=logger,
                               wandb_run=wandb_run,
                               epoch=0)
           
            
    #-----------------------------------------------------------------------------#
    # Train                                                                       #
    #-----------------------------------------------------------------------------#
    start_epoch = 0   
    best_score = 0
    

    for epoch in range(1, config.epochs+1):
        
        logger.info("%s[Epoch: %s]%s", 30*"-", epoch, 30*"-")

        if config.custom_sampling:
            train_dataloader.dataset.shuffle()
        with log_timer(logger, f"train_epoch_{epoch}", level=logging.INFO, sync_cuda=True), \
             WandbStepTimer(f"train_epoch_{epoch}", logger=logger, run=wandb_run, step=epoch, sync_cuda=True):
            train_loss = train_with_weight(config,
                               model,
                               dataloader=train_dataloader,
                               loss_function=loss_function_normal,
                               optimizer=optimizer,
                               scheduler=scheduler,
                               scaler=scaler,
                               with_weight=config.with_weight,
                               logger=logger,
                               wandb_run=wandb_run,
                               epoch=epoch)

            logger.info("第 %s 轮: 训练损失 = %.3f, 学习率 = %.6f", epoch,
                    train_loss, optimizer.param_groups[0]['lr'])
            safe_log(
                wandb_run,
                {
                    "train/loss": train_loss,
                    "train/lr": optimizer.param_groups[0]['lr'],
                },
                step=epoch,
            )
        
        # evaluate
        if (epoch % config.eval_every_n_epoch == 0 and epoch != 0) or epoch == config.epochs:
        
            logger.info("%s[Evaluate]%s", 30*"-", 30*"-")
            with log_timer(logger, f"evaluate_epoch_{epoch}", level=logging.INFO, sync_cuda=True), \
                 WandbStepTimer(f"evaluate_epoch_{epoch}", logger=logger, run=wandb_run, step=epoch, sync_cuda=True):
                r1_test = evaluate(config=config,
                                    model=model,
                                    query_loader=query_dataloader_test,
                                    gallery_loader=gallery_dataloader_test,
                                    query_list=query_img_list,
                                    gallery_list=gallery_img_list,
                                    pairs_dict=pairs_drone2sate_dict,
                                    ranks_list=[1, 5, 10],
                                    query_center_loc_xy_list=query_center_loc_xy_list,
                                    gallery_center_loc_xy_list=gallery_center_loc_xy_list,
                                    gallery_topleft_loc_xy_list=gallery_topleft_loc_xy_list,
                                    step_size=1000,
                                    cleanup=True,
                                    logger=logger,
                                    wandb_run=wandb_run,
                                    epoch=epoch)
                
            if r1_test > best_score or epoch == config.epochs:

                best_score = r1_test
                safe_log(
                    wandb_run,
                    {
                        "eval/best_recall@1": best_score * 100,
                    },
                    step=epoch,
                )

                with log_timer(logger, f"save_checkpoint_epoch_{epoch}", level=logging.INFO, sync_cuda=True), \
                     WandbStepTimer(f"save_checkpoint_epoch_{epoch}", logger=logger, run=wandb_run, step=epoch, sync_cuda=True):
                    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                        torch.save(model.module.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
                    else:
                        torch.save(model.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
                
    with log_timer(logger, "save_final_checkpoint", level=logging.INFO, sync_cuda=True), \
         WandbStepTimer("save_final_checkpoint", logger=logger, run=wandb_run, sync_cuda=True):
        if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
            torch.save(model.module.state_dict(), '{}/weights_end.pth'.format(model_path))
        else:
            torch.save(model.state_dict(), '{}/weights_end.pth'.format(model_path))

    logger.info("训练完成，最佳 Recall@1=%.4f", best_score)
    safe_log(
        wandb_run,
        {
            "final/best_recall@1": best_score * 100,
        },
        step=config.epochs,
    )
    finish_wandb(wandb_run, logger=logger)


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for gta.")

    parser.add_argument('--log_to_file', action='store_true', help='Log saving to file')

    parser.add_argument('--log_path', type=str, default=None, help='Log file path')

    parser.add_argument('--data_root', type=str, default='./data/GTA-UAV-data', help='Data root')

    parser.add_argument('--train_pairs_meta_file', type=str, default='cross-area-drone2sate-train.json', help='Training metafile path')
   
    parser.add_argument('--test_pairs_meta_file', type=str, default='cross-area-drone2sate-test.json', help='Test metafile path')

    parser.add_argument('--model', type=str, default='vit_base_patch16_rope_reg1_gap_256.sbb_in1k', help='Model architecture')

    parser.add_argument('--no_share_weights', action='store_true', help='Train without sharing wieghts')

    parser.add_argument('--freeze_layers', action='store_true', help='Freeze layers for training')

    parser.add_argument('--frozen_stages', type=int, nargs='+', default=[0,0,0,0], help='Frozen stages for training')

    parser.add_argument('--epochs', type=int, default=5, help='Epochs')

    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')

    parser.add_argument('--gpu_ids', type=parse_tuple, default=(0,1), help='GPU ID')

    parser.add_argument('--batch_size', type=int, default=40, help='Batch size')

    parser.add_argument('--checkpoint_start', type=str, default=None, help='Training from checkpoint')

    parser.add_argument('--train_mode', type=str, default='pos_semipos', help='Train with positive or positive+semi-positive pairs')

    parser.add_argument('--test_mode', type=str, default='pos', help='Test with positive pairs')

    parser.add_argument('--query_mode', type=str, default='D2S', help='Retrieval with drone to satellite')

    parser.add_argument('--train_with_recon', action='store_true', help='Train with reconstruction')

    parser.add_argument('--recon_weight', type=float, default=0.1, help='Loss weight for reconstruction')

    parser.add_argument('--train_in_group', action='store_true', help='Train in group')
    
    parser.add_argument('--group_len', type=int, default=2, help='Group length')

    parser.add_argument('--train_with_mix_data', action='store_true', help='Train with mix data')

    parser.add_argument('--loss_type', type=str, nargs='+', default=['part_slice', 'whole_slice'], help='Loss type for group train')

    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing value for loss')

    parser.add_argument('--with_weight', action='store_true', help='Train with weight')

    parser.add_argument('--k', type=float, default=5, help='weighted k')

    parser.add_argument('--no_custom_sampling', action='store_true', help='Train without custom sampling')
    
    parser.add_argument('--train_ratio', type=float, default=1.0, help='Train on ratio of data')
    parser.add_argument('--no_wandb', action='store_true', help='Disable Weights & Biases logging')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    config = Configuration()
    config.data_root = args.data_root
    config.train_pairs_meta_file = args.train_pairs_meta_file
    config.test_pairs_meta_file = args.test_pairs_meta_file
    config.log_to_file = args.log_to_file
    config.log_path = args.log_path
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.train_in_group = args.train_in_group
    config.train_with_recon = args.train_with_recon
    config.recon_weight = args.recon_weight
    config.group_len = args.group_len
    config.train_with_mix_data = args.train_with_mix_data
    config.loss_type = args.loss_type
    config.gpu_ids = args.gpu_ids
    config.label_smoothing = args.label_smoothing
    config.with_weight = args.with_weight
    config.k = args.k
    config.checkpoint_start = args.checkpoint_start
    config.model = args.model
    config.lr = args.lr
    config.share_weights = not(args.no_share_weights)
    config.custom_sampling = not(args.no_custom_sampling)
    config.freeze_layers = args.freeze_layers
    config.frozen_stages = args.frozen_stages
    config.train_mode = args.train_mode
    config.test_mode = args.test_mode
    config.query_mode = args.query_mode
    config.train_ratio = args.train_ratio
    config.use_wandb = not(args.no_wandb)

    train_script(config)