import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Add project paths
game4loc_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Game4Loc")
sys.path.append(game4loc_dir)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game4loc.dataset.gta import GTADatasetTrain, get_transforms
from train_gta import Configuration
from geometry.pose_grid import pose_to_grid
from geometry.pose_attention import distortion_to_attention, resize_attention_map

def denormalize_image(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize image tensor to numpy array [0, 1] for visualization."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    img = img_tensor * std + mean
    img = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).numpy()

def main():
    print("==========================================")
    print("  Pose Attention Prior Batch Verification  ")
    print("==========================================\n")
    
    out_dir = "debug_outputs/pose_attention"
    os.makedirs(out_dir, exist_ok=True)
    
    config = Configuration()
    config.data_root = "Game4Loc/data/GTA-UAV-data"
    batch_size = 4
    
    # 1. Load real data
    print(">>> 1. Loading real batch from GTA-UAV dataset...")
    val_transforms, train_sat_transforms, train_drone_transforms = \
        get_transforms((384, 384), sat_rot=False)
        
    train_dataset = GTADatasetTrain(data_root=config.data_root,
                                    pairs_meta_file=config.train_pairs_meta_file,
                                    transforms_query=train_drone_transforms,
                                    transforms_gallery=train_sat_transforms)
                                    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    # Get a batch
    batch = next(iter(train_loader))
    query_imgs, _, _, query_poses = batch
    
    B, C, H, W = query_imgs.shape
    fov_deg = 70.0 # Default FOV
    
    # Target ViT patch grid size (384 // 16 = 24)
    target_hw = (H // 16, W // 16)
    
    print(f"Batch size: {B}")
    print(f"Image shape: {H}x{W}")
    print(f"Target patch grid shape: {target_hw[0]}x{target_hw[1]}")
    
    # Process each sample in the batch
    print("\n>>> 2. Processing batch samples...")
    
    fig, axs = plt.subplots(B, 4, figsize=(16, 4*B))
    
    # Collect attention maps to verify they are different
    attn_maps_collected = []
    
    for i in range(B):
        img = query_imgs[i]
        pose = query_poses[i].numpy()
        height, pitch, roll = pose[0], pose[1], pose[2]
        
        print(f"\n--- Sample {i} ---")
        print(f"Pose: height={height:.2f}, pitch={pitch:.2f}, roll={roll:.2f}")
        
        # A. Get geometry base
        grid_xy, distortion, valid_mask = pose_to_grid(H, W, height, pitch, roll, fov_deg)
        
        valid_ratio = valid_mask.mean()
        print(f"Valid ratio: {valid_ratio:.4f}")
        
        # B. Generate attention prior
        attn_map = distortion_to_attention(distortion, valid_mask, mode="inverse", floor=0.3, alpha=1.0)
        attn_maps_collected.append(attn_map)
        
        attn_min = attn_map.min()
        attn_max = attn_map.max()
        attn_mean = attn_map.mean()
        print(f"Attention stats - min: {attn_min:.4f}, max: {attn_max:.4f}, mean: {attn_mean:.4f}")
        
        # C. Resize to feature map size
        resized_attn = resize_attention_map(attn_map, target_hw, mode="bilinear")
        
        # Visualization
        # 1. Image
        img_vis = denormalize_image(img)
        axs[i, 0].imshow(img_vis)
        axs[i, 0].set_title(f"Sample {i}: Image")
        axs[i, 0].axis('off')
        
        # 2. Distortion Heatmap (log)
        dist_plot = np.log10(distortion + 1e-6)
        dist_plot[~valid_mask] = np.nan
        im1 = axs[i, 1].imshow(dist_plot, cmap='viridis')
        axs[i, 1].set_title(f"Distortion (log)\npitch={pitch:.1f}, roll={roll:.1f}")
        fig.colorbar(im1, ax=axs[i, 1])
        axs[i, 1].axis('off')
        
        # 3. Attention Prior
        im2 = axs[i, 2].imshow(attn_map, cmap='magma', vmin=0.0, vmax=1.0)
        axs[i, 2].set_title(f"Attention Prior\n[{attn_min:.2f}, {attn_max:.2f}]")
        fig.colorbar(im2, ax=axs[i, 2])
        axs[i, 2].axis('off')
        
        # 4. Resized Attention Prior
        im3 = axs[i, 3].imshow(resized_attn, cmap='magma', vmin=0.0, vmax=1.0)
        axs[i, 3].set_title(f"Resized Attn\n{target_hw[0]}x{target_hw[1]}")
        fig.colorbar(im3, ax=axs[i, 3])
        axs[i, 3].axis('off')
        
    plt.tight_layout()
    out_path = os.path.join(out_dir, "batch_attention_verification.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved visualization to {out_path}")
    
    # Verify attention maps are different
    print("\n>>> 3. Verifying batch diversity...")
    is_diff_0_1 = not np.allclose(attn_maps_collected[0], attn_maps_collected[1])
    is_diff_1_2 = not np.allclose(attn_maps_collected[1], attn_maps_collected[2])
    
    if is_diff_0_1 and is_diff_1_2:
        print("[SUCCESS] Attention maps are different across batch samples!")
    else:
        print("[WARNING] Some attention maps in the batch are identical!")
        
    print("\n[CONCLUSION] Pose -> Attention Prior pipeline is ready!")

if __name__ == "__main__":
    main()
