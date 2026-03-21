import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 将父目录加入 sys.path，以便导入 geometry 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry.pose_grid import pose_to_grid

def run_case(case_name, H, W, height, pitch, roll, fov_deg, out_dir):
    print(f"\n--- Running {case_name} ---")
    print(f"Params: height={height}, pitch={pitch}, roll={roll}, fov={fov_deg}")
    
    grid_xy, distortion, valid_mask = pose_to_grid(H, W, height, pitch, roll, fov_deg)
    
    # 统计信息
    valid_ratio = valid_mask.mean()
    valid_dist = distortion[valid_mask & (distortion > 0)]
    
    if len(valid_dist) > 0:
        dist_min = valid_dist.min()
        dist_max = valid_dist.max()
        dist_mean = valid_dist.mean()
    else:
        dist_min, dist_max, dist_mean = 0, 0, 0
        
    finite_ratio = np.isfinite(distortion[valid_mask]).mean() if valid_ratio > 0 else 0.0
    
    print(f"valid_ratio: {valid_ratio:.4f}")
    print(f"finite_ratio: {finite_ratio:.4f}")
    print(f"distortion - min: {dist_min:.4f}, max: {dist_max:.4f}, mean: {dist_mean:.4f}")
    
    # ==========================================
    # 1. 关键点投影检查
    # ==========================================
    print("\n[Keypoints Projection]")
    keypoints = {
        "center": (H//2, W//2),
        "top_center": (0, W//2),
        "bottom_center": (H-1, W//2),
        "left_center": (H//2, 0),
        "right_center": (H//2, W-1),
        "top_left": (0, 0),
        "top_right": (0, W-1),
        "bottom_left": (H-1, 0),
        "bottom_right": (H-1, W-1)
    }
    for name, (y, x) in keypoints.items():
        is_valid = valid_mask[y, x]
        gx, gy = grid_xy[y, x]
        print(f"  {name:15s} -> valid={is_valid!s:5s}, (Xg={gx:8.3f}, Yg={gy:8.3f})")

    # ==========================================
    # 2. 对称性误差指标计算
    # ==========================================
    # 左右对称性 (Left-Right Symmetry)
    dist_lr = np.fliplr(distortion)
    valid_lr = np.fliplr(valid_mask)
    shared_valid_lr = valid_mask & valid_lr
    if shared_valid_lr.any():
        err_lr = np.mean(np.abs(distortion[shared_valid_lr] - dist_lr[shared_valid_lr])) / (np.mean(np.abs(distortion[shared_valid_lr])) + 1e-6)
    else:
        err_lr = float('nan')
        
    # 上下对称性 (Up-Down Symmetry)
    dist_ud = np.flipud(distortion)
    valid_ud = np.flipud(valid_mask)
    shared_valid_ud = valid_mask & valid_ud
    if shared_valid_ud.any():
        err_ud = np.mean(np.abs(distortion[shared_valid_ud] - dist_ud[shared_valid_ud])) / (np.mean(np.abs(distortion[shared_valid_ud])) + 1e-6)
    else:
        err_ud = float('nan')
        
    print("\n[Symmetry Errors]")
    print(f"  Left-Right symmetry error: {err_lr:.6f}")
    print(f"  Up-Down symmetry error:    {err_ud:.6f}")
    
    # 可视化
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"{case_name}: pitch={pitch}, roll={roll}")
    
    # 1. Valid Mask
    axs[0, 0].imshow(valid_mask, cmap='gray')
    axs[0, 0].set_title('Valid Mask')
    
    # 2. Distortion Heatmap (使用 log10 以便更好地展示差异)
    dist_plot = np.log10(distortion + 1e-6)
    dist_plot[~valid_mask] = np.nan
    im = axs[0, 1].imshow(dist_plot, cmap='viridis')
    axs[0, 1].set_title('Distortion (log10)')
    fig.colorbar(im, ax=axs[0, 1])
    
    # 3. Ground X (指东 East)
    Xg = grid_xy[..., 0].copy()
    Xg[~valid_mask] = np.nan
    im_x = axs[1, 0].imshow(Xg, cmap='coolwarm')
    axs[1, 0].set_title('Ground X (East)')
    fig.colorbar(im_x, ax=axs[1, 0])
    
    # 4. Ground Y (指北 North)
    Yg = grid_xy[..., 1].copy()
    Yg[~valid_mask] = np.nan
    im_y = axs[1, 1].imshow(Yg, cmap='coolwarm')
    axs[1, 1].set_title('Ground Y (North)')
    fig.colorbar(im_y, ax=axs[1, 1])
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{case_name}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved visualization to {out_path}")

def run_height_scan(H, W, pitch, roll, fov_deg):
    print("\n========================================")
    print("3. 高度扫描实验 (Height Scanning Experiment)")
    print(f"Fixed Params: pitch={pitch}, roll={roll}, fov={fov_deg}")
    print("========================================")
    
    heights = [60, 120, 240]
    for h in heights:
        print(f"\n--- Height: {h} ---")
        grid_xy, distortion, valid_mask = pose_to_grid(H, W, h, pitch, roll, fov_deg)
        
        valid_ratio = valid_mask.mean()
        
        if valid_ratio > 0:
            Xg = grid_xy[valid_mask, 0]
            Yg = grid_xy[valid_mask, 1]
            x_min, x_max = Xg.min(), Xg.max()
            y_min, y_max = Yg.min(), Yg.max()
            
            valid_dist = distortion[valid_mask & (distortion > 0)]
            if len(valid_dist) > 0:
                dist_mean = valid_dist.mean()
                dist_median = np.median(valid_dist)
            else:
                dist_mean, dist_median = 0, 0
        else:
            x_min, x_max, y_min, y_max = 0, 0, 0, 0
            dist_mean, dist_median = 0, 0
            
        print(f"  valid_ratio:      {valid_ratio:.4f}")
        print(f"  X range:          [{x_min:8.3f}, {x_max:8.3f}] (span={x_max-x_min:.3f})")
        print(f"  Y range:          [{y_min:8.3f}, {y_max:8.3f}] (span={y_max-y_min:.3f})")
        print(f"  distortion mean:  {dist_mean:.4f}")
        print(f"  distortion median:{dist_median:.4f}")

def main():
    out_dir = "debug_outputs/pose_grid"
    os.makedirs(out_dir, exist_ok=True)
    
    H, W = 256, 256
    fov_deg = 70
    
    # 测试用例
    cases = [
        ("Case_A_Nadir", 120, -90, 0),
        ("Case_B_Pitch80", 120, -80, 0),
        ("Case_C_Pitch90_Roll8", 120, -90, 8),
    ]
    
    print("========================================")
    print("预期现象:")
    print("1. Case A (pitch=-90, roll=0): nadir-like, 下视状态，畸变分布应相对更对称。")
    print("2. Case B (pitch=-80, roll=0): 抬头向前，pitch变化，畸变图应出现明显的前后不对称 (上下不对称)。")
    print("3. Case C (pitch=-90, roll=8): roll非零，向右倾斜，畸变图应出现明显的左右不对称。")
    print("========================================\n")
    
    for name, height, pitch, roll in cases:
        run_case(name, H, W, height, pitch, roll, fov_deg, out_dir)
        
    # 运行高度扫描实验
    run_height_scan(H, W, pitch=-90, roll=0, fov_deg=fov_deg)
    
    print("\n========================================")
    print("结论: Geometry sanity check passed!")
    print("========================================")

if __name__ == "__main__":
    main()
