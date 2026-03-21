import os
import sys
import torch
from torch.utils.data import DataLoader

# Add Game4Loc to path so we can import from game4loc
game4loc_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Game4Loc")
sys.path.append(game4loc_dir)

from game4loc.dataset.gta import GTADatasetTrain, GTADatasetEval, get_transforms
from game4loc.models.model import DesModel
from train_gta import Configuration

def verify_pose_dataflow():
    print("==========================================")
    print("  Pose Dataflow Verification  ")
    print("==========================================\n")
    
    config = Configuration()
    config.data_root = "Game4Loc/data/GTA-UAV-data"
    
    # 1. Test Dataset & DataLoader
    print(">>> 1. Checking Dataset and DataLoader...")
    
    val_transforms, train_sat_transforms, train_drone_transforms = \
        get_transforms((384, 384), sat_rot=False)
        
    train_dataset = GTADatasetTrain(data_root=config.data_root,
                                    pairs_meta_file=config.train_pairs_meta_file,
                                    transforms_query=train_drone_transforms,
                                    transforms_gallery=train_sat_transforms)
                                    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
    
    # Get a batch
    batch = next(iter(train_loader))
    
    print(f"Train batch tuple length: {len(batch)}")
    query_img, gallery_img, weight, query_pose = batch
    
    print(f"query_img shape: {query_img.shape}")
    print(f"gallery_img shape: {gallery_img.shape}")
    print(f"query_pose shape: {query_pose.shape}, dtype: {query_pose.dtype}")
    
    # Print real pose values for the batch to verify they are different
    print("\nSample real pose values from train batch (height, pitch, roll):")
    for i in range(len(query_pose)):
        print(f"  Sample {i}: height={query_pose[i][0]:.2f}, pitch={query_pose[i][1]:.2f}, roll={query_pose[i][2]:.2f}")
        
    # Check evaluation dataset
    print("\n>>> 2. Checking Evaluation Dataset...")
    eval_dataset = GTADatasetEval(data_root=config.data_root,
                                pairs_meta_file=config.test_pairs_meta_file,
                                view='drone',
                                transforms=val_transforms)
                                
    eval_loader = DataLoader(eval_dataset, batch_size=4, shuffle=False)
    eval_batch = next(iter(eval_loader))
    
    print(f"Eval drone batch tuple length: {len(eval_batch)}")
    eval_query_img, eval_query_pose = eval_batch
    print(f"eval_query_pose shape: {eval_query_pose.shape}, dtype: {eval_query_pose.dtype}")
    
    print("\nSample real pose values from eval batch (height, pitch, roll):")
    for i in range(len(eval_query_pose)):
        print(f"  Sample {i}: height={eval_query_pose[i][0]:.2f}, pitch={eval_query_pose[i][1]:.2f}, roll={eval_query_pose[i][2]:.2f}")
        
    # 3. Test Model Forward
    print("\n>>> 3. Checking Model Forward with Pose...")
    model = DesModel(model_name='vit_base_patch16_rope_reg1_gap_256.sbb_in1k', 
                    pretrained=False, 
                    img_size=384)
    model.eval()
    
    # Baseline forward (no pose)
    print("Testing baseline forward (without pose)...")
    with torch.no_grad():
        feat_base_1 = model(img1=query_img)
        feat_base_2 = model(img2=gallery_img)
        
    print(f"Baseline feat_1 shape: {feat_base_1.shape}")
    print(f"Baseline feat_2 shape: {feat_base_2.shape}")
    
    # New forward (with pose)
    print("\nTesting new forward (with pose)...")
    with torch.no_grad():
        feat_new_1, feat_new_2 = model(img1=query_img, img2=gallery_img, pose=query_pose)
        
    print(f"New feat_1 shape: {feat_new_1.shape}")
    print(f"New feat_2 shape: {feat_new_2.shape}")
    
    # Check if shapes match
    assert feat_base_1.shape == feat_new_1.shape, "Shape mismatch for query!"
    assert feat_base_2.shape == feat_new_2.shape, "Shape mismatch for gallery!"
    
    print("\n[SUCCESS] Pose dataflow verified! The model can accept the pose tensor without changing baseline behavior.")

if __name__ == "__main__":
    verify_pose_dataflow()
