import torch
import torch.nn as nn

class PoseAttentionGate(nn.Module):
    """
    无参数的 Pose Attention Gating 模块。
    基于物理先验计算出的 attention map 对图像特征进行门控。
    
    该模块设计为与 ViT 的 patch token 进行交互：
    1. 不 gate CLS token
    2. 只对 patch tokens 按照其空间位置乘以对应权重的 attention prior
    """
    def __init__(self, target_hw=None):
        """
        参数:
            target_hw: (Hf, Wf), 特征图的期望空间尺寸。对于 ViT-B (patch=16, img=384)，应为 (24, 24)。
                       如果不提供，在 forward 时会尝试从特征形状自动推断。
        """
        super().__init__()
        self.target_hw = target_hw

    def forward(self, feature_map, attn_map, mode="multiplicative", lambda_val=0.5):
        """
        参数:
            feature_map: 
                对于 ViT: [B, N, C] 或 [B, N+1, C] (含 cls token)
                对于 CNN: [B, C, H, W]
            attn_map: [B, Hf, Wf] (已经 resize 好的 attention prior)
            mode: "multiplicative" 或 "residual"
            lambda_val: residual 模式下的控制强度 [0, 1]
            
        返回:
            gated_feature: 与 feature_map 形状完全一致
        """
        if attn_map is None:
            return feature_map

        B = feature_map.shape[0]
        
        # 1. 判断是 ViT (3D tensor) 还是 CNN (4D tensor)
        if feature_map.dim() == 3:
            # ViT 模式: [B, L, C]
            B, L, C = feature_map.shape
            
            # 推断是否存在 cls token
            # 如果 L = Hf * Wf，则无 cls token
            # 如果 L = Hf * Wf + 1，则有 1 个 cls token
            # 如果 L = Hf * Wf + N_cls，则有 N_cls 个 cls token
            
            if self.target_hw is not None:
                Hf, Wf = self.target_hw
                num_patches = Hf * Wf
            else:
                # 尝试从 L 自动推断，假设是正方形网格
                # 例如 24*24 = 576, 如果 L=577，则 num_patches=576
                num_patches = attn_map.shape[1] * attn_map.shape[2]
                
            num_cls_tokens = L - num_patches
            
            if num_cls_tokens < 0:
                raise ValueError(f"Feature sequence length {L} is smaller than expected patch count {num_patches}")
                
            # 分离 cls tokens 和 patch tokens
            if num_cls_tokens > 0:
                cls_tokens = feature_map[:, :num_cls_tokens, :]
                patch_tokens = feature_map[:, num_cls_tokens:, :]
            else:
                patch_tokens = feature_map
                
            # 将 attention map 展平并对齐维度: [B, Hf, Wf] -> [B, Hf*Wf, 1]
            attn_flat = attn_map.view(B, num_patches, 1)
            
            # Gating (Broadcast)
            if mode == "multiplicative":
                gated_patch_tokens = patch_tokens * attn_flat
            elif mode == "residual":
                # x' = x * (1 + lambda * (A - 1))
                gated_patch_tokens = patch_tokens * (1.0 + lambda_val * (attn_flat - 1.0))
            else:
                raise ValueError(f"Unknown gate mode: {mode}")
            
            # 拼接回原状
            if num_cls_tokens > 0:
                gated_feature = torch.cat([cls_tokens, gated_patch_tokens], dim=1)
            else:
                gated_feature = gated_patch_tokens
                
            return gated_feature
            
        elif feature_map.dim() == 4:
            # CNN 模式: [B, C, H, W]
            # 确保 attn_map 匹配特征图的 H 和 W
            B, C, Hf, Wf = feature_map.shape
            
            if attn_map.shape[1] != Hf or attn_map.shape[2] != Wf:
                import torch.nn.functional as F
                attn_map = F.interpolate(attn_map.unsqueeze(1), size=(Hf, Wf), mode='bilinear').squeeze(1)
                
            # 扩展维度: [B, Hf, Wf] -> [B, 1, Hf, Wf]
            attn_expanded = attn_map.unsqueeze(1)
            
            # Gating
            if mode == "multiplicative":
                gated_feature = feature_map * attn_expanded
            elif mode == "residual":
                gated_feature = feature_map * (1.0 + lambda_val * (attn_expanded - 1.0))
            else:
                raise ValueError(f"Unknown gate mode: {mode}")
                
            return gated_feature
            
        else:
            raise ValueError(f"Unsupported feature_map shape: {feature_map.shape}")


def compute_feature_energy(feature_map):
    """
    计算特征图的空间能量分布，用于可视化和 debug。
    
    参数:
        feature_map: [B, L, C] (ViT) 或 [B, C, H, W] (CNN)
        
    返回:
        energy_map: [B, Hf, Wf]，每个空间位置的 L2 norm
    """
    if feature_map.dim() == 3:
        B, L, C = feature_map.shape
        # 假设正方形网格并排除 1 个 cls token
        num_patches = L
        if L == 24*24 + 1:
            num_patches = 24*24
            patch_tokens = feature_map[:, 1:, :]
        else:
            # 尝试推断
            hw = int(L**0.5)
            if hw*hw == L:
                patch_tokens = feature_map
            elif hw*hw + 1 == L:
                patch_tokens = feature_map[:, 1:, :]
                num_patches = hw*hw
            else:
                # 给出一个通用的后备处理
                patch_tokens = feature_map[:, (L - int((L-1)**0.5)**2):, :]
                num_patches = patch_tokens.shape[1]
                
        # 计算 L2 norm
        energy = torch.norm(patch_tokens, dim=-1) # [B, num_patches]
        
        # reshape 回 2D
        H = int(num_patches ** 0.5)
        W = num_patches // H
        
        energy_map = energy.view(B, H, W)
        return energy_map
        
    elif feature_map.dim() == 4:
        # [B, C, H, W] -> 计算通道维度的 L2 norm
        energy_map = torch.norm(feature_map, dim=1) # [B, H, W]
        return energy_map
    else:
        return None
