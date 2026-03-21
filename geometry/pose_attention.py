import numpy as np
import torch
import torch.nn.functional as F

def normalize_distortion(distortion, valid_mask, method="robust", q_low=1.0, q_high=99.0):
    """
    将 distortion 归一化到近似稳定的 [0, 1] 范围。
    只在 valid_mask 区域内进行统计。
    
    参数:
        distortion: np.ndarray, 形状为 [H, W]，或者 torch.Tensor
        valid_mask: np.ndarray, 形状为 [H, W]，或者 torch.Tensor
        method: str, 归一化方法，目前支持 "robust"
        q_low, q_high: float, robust 方法下的分位数阈值
        
    返回:
        D_norm: 归一化后的 distortion，无效区域将保留原始的 0 或被置 0。
    """
    is_tensor = isinstance(distortion, torch.Tensor)
    
    if is_tensor:
        D = distortion.clone()
        mask = valid_mask.bool()
        
        if not mask.any():
            return torch.zeros_like(D)
            
        valid_vals = D[mask]
        
        if method == "robust":
            # PyTorch 中求分位数要求 float32 或 float64
            valid_vals_f = valid_vals.to(torch.float32)
            vmin = torch.quantile(valid_vals_f, q_low / 100.0)
            vmax = torch.quantile(valid_vals_f, q_high / 100.0)
            
            # 避免除以 0
            if vmax - vmin < 1e-6:
                D_norm = torch.zeros_like(D)
                D_norm[mask] = 0.5  # 恒定值
            else:
                # 裁剪并归一化
                D_norm = torch.clamp((D - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
                
            D_norm[~mask] = 0.0
            return D_norm
        else:
            raise ValueError(f"Unknown normalization method: {method}")
            
    else:
        D = distortion.copy()
        mask = valid_mask.astype(bool)
        
        if not mask.any():
            return np.zeros_like(D)
            
        valid_vals = D[mask]
        
        if method == "robust":
            vmin = np.percentile(valid_vals, q_low)
            vmax = np.percentile(valid_vals, q_high)
            
            if vmax - vmin < 1e-6:
                D_norm = np.zeros_like(D)
                D_norm[mask] = 0.5
            else:
                D_norm = np.clip((D - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
                
            D_norm[~mask] = 0.0
            return D_norm
        else:
            raise ValueError(f"Unknown normalization method: {method}")


def distortion_to_attention(distortion, valid_mask, mode="inverse", floor=0.3, alpha=1.0):
    """
    将 distortion 转换为 attention prior。
    核心逻辑：distortion 越大，attention 权重越小。
    
    参数:
        distortion: np.ndarray 或 torch.Tensor, [H, W]
        valid_mask: np.ndarray 或 torch.Tensor, [H, W]
        mode: str, "inverse" 模式
        floor: float, attention 的下界，同时也作为 invalid 区域的默认值
        alpha: float, 调节函数陡峭程度
        
    返回:
        A: attention prior map，[H, W]
    """
    is_tensor = isinstance(distortion, torch.Tensor)
    
    # 1. 归一化
    D_norm = normalize_distortion(distortion, valid_mask, method="robust")
    
    # 2. 转换为 Attention
    if is_tensor:
        mask = valid_mask.bool()
        A = torch.full_like(D_norm, fill_value=floor)
        
        if mode == "inverse":
            A_raw = 1.0 / (1.0 + alpha * D_norm)
            # 缩放到 [floor, 1.0]
            A_valid = floor + (1.0 - floor) * A_raw
            A[mask] = A_valid[mask]
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
    else:
        mask = valid_mask.astype(bool)
        A = np.full_like(D_norm, fill_value=floor)
        
        if mode == "inverse":
            A_raw = 1.0 / (1.0 + alpha * D_norm)
            A_valid = floor + (1.0 - floor) * A_raw
            A[mask] = A_valid[mask]
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
    return A


def resize_attention_map(attn_map, target_hw, mode="bilinear"):
    """
    将 [H, W] 的 attention map resize 到 target_hw [Hf, Wf]
    
    参数:
        attn_map: [H, W]
        target_hw: tuple (Hf, Wf)
        mode: resize 模式
        
    返回:
        resized_map: [Hf, Wf]
    """
    is_tensor = isinstance(attn_map, torch.Tensor)
    
    if is_tensor:
        # F.interpolate 需要 [B, C, H, W]
        x = attn_map.unsqueeze(0).unsqueeze(0).to(torch.float32)
        x_resized = F.interpolate(x, size=target_hw, mode=mode, align_corners=False)
        return x_resized.squeeze(0).squeeze(0)
    else:
        import cv2
        # cv2.resize 需要 target_hw = (Wf, Hf)
        target_size_cv = (target_hw[1], target_hw[0])
        if mode == "bilinear":
            interp = cv2.INTER_LINEAR
        elif mode == "nearest":
            interp = cv2.INTER_NEAREST
        else:
            interp = cv2.INTER_LINEAR
            
        x_resized = cv2.resize(attn_map.astype(np.float32), target_size_cv, interpolation=interp)
        return x_resized
