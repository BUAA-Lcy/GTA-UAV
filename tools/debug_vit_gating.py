import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

# Add project paths
game4loc_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Game4Loc")
sys.path.append(game4loc_dir)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game4loc.dataset.gta import GTADatasetTrain, get_transforms
from train_gta import Configuration
from game4loc.models.model import DesModel

def main():
    print("==========================================")
    print("  ViT Gating Verification  ")
    print("==========================================\n")
    
    config = Configuration()
    config.data_root = "Game4Loc/data/GTA-UAV-data"
    batch_size = 2
    
    # 1. Load real data
    print(">>> 1. Loading real batch from GTA-UAV dataset...")
    val_transforms, train_sat_transforms, train_drone_transforms = \
        get_transforms((384, 384), sat_rot=False)
        
    train_dataset = GTADatasetTrain(data_root=config.data_root,
                                    pairs_meta_file=config.train_pairs_meta_file,
                                    transforms_query=train_drone_transforms,
                                    transforms_gallery=train_sat_transforms)
                                    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    batch = next(iter(train_loader))
    query_imgs, gallery_imgs, _, query_poses = batch
    
    # 2. Setup Model
    print("\n>>> 2. Initializing ViT Model...")
    model = DesModel(model_name='vit_base_patch16_rope_reg1_gap_256.sbb_in1k', 
                    pretrained=False, 
                    img_size=384)
    model.eval()
    
    # Ensure reproducibility
    torch.manual_seed(42)
    
    # 3. Test Baseline (Switch OFF)
    print("\n>>> 3. Testing Baseline (use_pose_attention = False)")
    model.use_pose_attention = False
    with torch.no_grad():
        feat_base_uav, feat_base_sat = model(img1=query_imgs, img2=gallery_imgs, pose=query_poses)
    
    print(f"  feat_base_uav shape: {feat_base_uav.shape}")
    
    # 4. Test Attention Gating (Switch ON)
    print("\n>>> 4. Testing Pose Attention Gating (use_pose_attention = True)")
    model.use_pose_attention = True
    with torch.no_grad():
        feat_gate_uav, feat_gate_sat = model(img1=query_imgs, img2=gallery_imgs, pose=query_poses)
        
    print(f"  feat_gate_uav shape: {feat_gate_uav.shape}")
    
    # 5. Verification Checks
    print("\n>>> 5. Verifying Results...")
    
    # Check 1: Shapes should be identical
    assert feat_base_uav.shape == feat_gate_uav.shape, "Shape mismatch between baseline and gated features!"
    print("  [Pass] Gated features maintain original shape.")
    
    # Check 2: Satellite branch should be completely unaffected
    sat_diff = torch.max(torch.abs(feat_base_sat - feat_gate_sat)).item()
    print(f"  Max difference in Satellite branch: {sat_diff}")
    assert sat_diff < 1e-6, "Satellite branch was unexpectedly modified!"
    print("  [Pass] Satellite branch is unaffected.")
    
    # Check 3: UAV branch should be modified
    uav_diff = torch.max(torch.abs(feat_base_uav - feat_gate_uav)).item()
    print(f"  Max difference in UAV branch: {uav_diff}")
    assert uav_diff > 1e-4, "UAV branch was not modified by gating!"
    print("  [Pass] UAV branch features were successfully gated.")
    
    # Check 4: Energy stats
    base_energy = torch.norm(feat_base_uav, dim=-1).mean().item()
    gate_energy = torch.norm(feat_gate_uav, dim=-1).mean().item()
    print(f"  UAV mean energy: Baseline={base_energy:.4f}, Gated={gate_energy:.4f}")
    
    print("\n[SUCCESS] ViT Forward Gating implementation is verified!")

if __name__ == "__main__":
    main()
