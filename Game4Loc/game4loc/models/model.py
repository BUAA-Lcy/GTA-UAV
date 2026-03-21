import torch
import timm
import numpy as np
import torch.nn as nn
from PIL import Image
from urllib.request import urlopen
from thop import profile
import sys
import os

# Add geometry path to sys.path to ensure absolute imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from geometry.pose_grid import pose_to_grid
from geometry.pose_attention import distortion_to_attention, resize_attention_map
from game4loc.models.modules.pose_gate import PoseAttentionGate


class MLP(nn.Module):
    def __init__(self, input_size=2048, hidden_size=512, output_size=2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class DesModel(nn.Module):

    def __init__(self, 
                 model_name='vit',
                 pretrained=True,
                 img_size=384,
                 share_weights=True,
                 train_with_recon=False,
                 train_with_offset=False,
                 model_hub='timm'):
                 
        super(DesModel, self).__init__()
        self.share_weights = share_weights
        self.model_name = model_name
        self.img_size = img_size
        if share_weights:
            if "vit" in model_name or "swin" in model_name:
                # automatically change interpolate pos-encoding to img_size
                self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size) 
            else:
                self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        else:
            if "vit" in model_name or "swin" in model_name:
                self.model1 = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size)
                self.model2 = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size) 
            else:
                self.model1 = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
                self.model2 = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        if train_with_offset:
            self.MLP = MLP()
        
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        # Pose Attention Gating Module
        # By default it's disabled. Enable it by setting use_pose_attention=True.
        self.use_pose_attention = False
        self.pose_attn_floor = 0.3
        self.pose_gate_mode = "multiplicative"
        self.pose_gate_lambda = 0.5
        self.pose_gate_insert_stage = "pre_blocks"
        self.pose_gate = PoseAttentionGate()
        self.fov_deg = 70.0  # Default FOV for GTA-UAV
        
    def _apply_pose_attention(self, model_instance, img, pose):
        """
        在 ViT 的 patch embedding 之后，transformer blocks 之前应用 Pose Attention Gating。
        目前仅支持 timm 的 VisionTransformer。
        """
        if not self.use_pose_attention or pose is None:
            return model_instance(img)
            
        if not isinstance(model_instance, timm.models.vision_transformer.VisionTransformer) and not hasattr(model_instance, 'patch_embed'):
            print("Warning: Pose attention is currently only implemented for VisionTransformer. Skipping gating.")
            return model_instance(img)
            
        B, C, H, W = img.shape
        
        # 1. 生成 batch 的 attention prior maps
        attn_maps = []
        # ViT-Base 对应的 patch grid size (img_size // patch_size)
        patch_size = model_instance.patch_embed.patch_size[0]
        target_hw = (H // patch_size, W // patch_size)
        
        for i in range(B):
            h, p, r = pose[i][0].item(), pose[i][1].item(), pose[i][2].item()
            _, distortion, valid_mask = pose_to_grid(H, W, h, p, r, self.fov_deg)
            attn_map_np = distortion_to_attention(distortion, valid_mask, mode="inverse", floor=self.pose_attn_floor, alpha=1.0)
            resized_attn = resize_attention_map(attn_map_np, target_hw, mode="bilinear")
            attn_maps.append(torch.from_numpy(resized_attn))
            
        # [B, Hf, Wf]
        attn_tensor = torch.stack(attn_maps).to(img.device)
        
        # 2. 拦截并修改 ViT forward
        # a. Patch Embedding
        x = model_instance.patch_embed(img)
        
        # Handle _pos_embed depending on timm version
        if hasattr(model_instance, '_pos_embed'):
            x = model_instance._pos_embed(x)
        else:
            if model_instance.cls_token is not None:
                x = torch.cat((model_instance.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + model_instance.pos_embed
            x = model_instance.pos_drop(x)
            
        # Apply Pose Attention Gating logic
        def apply_gate(feat):
            feat_before_gate = feat[0].clone() if isinstance(feat, tuple) else feat.clone()
            
            if isinstance(feat, tuple):
                feat_main = feat[0]
                rot_pos_emb = feat[1:]
                feat_main = self.pose_gate(feat_main, attn_tensor, mode=self.pose_gate_mode, lambda_val=self.pose_gate_lambda)
                feat = (feat_main,) + rot_pos_emb
            else:
                feat = self.pose_gate(feat, attn_tensor, mode=self.pose_gate_mode, lambda_val=self.pose_gate_lambda)
                
            feat_after_gate = feat[0].clone() if isinstance(feat, tuple) else feat.clone()
            
            if np.random.rand() < 0.01:
                cls_diff = torch.max(torch.abs(feat_before_gate[:, 0, :] - feat_after_gate[:, 0, :])).item()
                patch_diff = torch.max(torch.abs(feat_before_gate[:, 1:, :] - feat_after_gate[:, 1:, :])).item()
                attn_min = attn_tensor.min().item()
                attn_max = attn_tensor.max().item()
                attn_mean = attn_tensor.mean().item()
                print(f"\n[Pose Gate Verification] mode: {self.pose_gate_mode}, stage: {self.pose_gate_insert_stage}, "
                      f"cls diff: {cls_diff:.6f}, patch diff: {patch_diff:.6f}, "
                      f"attn: [{attn_min:.2f}, {attn_max:.2f}], mean: {attn_mean:.2f}")
                if cls_diff > 1e-5:
                    print("WARNING: CLS token was modified by pose gate!")
            return feat

        if self.pose_gate_insert_stage == "pre_blocks":
            x = apply_gate(x)

        # c. 继续后续的 Forward
        # Some timm models like EVA return a tuple from _pos_embed where the second element is RoPE.
        # But their blocks take only the main tensor or take kwargs.
        # We need to adapt based on what x is.
        if isinstance(x, tuple):
            x_main = x[0]
            rope = x[1] if len(x) > 1 else None
        else:
            x_main = x
            rope = None

        if hasattr(model_instance, 'norm_pre') and model_instance.norm_pre is not None:
            x_main = model_instance.norm_pre(x_main)
            
        if isinstance(model_instance.blocks, nn.ModuleList) or isinstance(model_instance.blocks, nn.Sequential):
            for i, blk in enumerate(model_instance.blocks):
                if rope is not None and 'rope' in blk.forward.__code__.co_varnames:
                    x_main = blk(x_main, rope=rope)
                else:
                    x_main = blk(x_main)
                    
                if self.pose_gate_insert_stage == f"after_block{i+1}":
                    if rope is not None:
                        x = apply_gate((x_main, rope))
                        x_main = x[0]
                    else:
                        x_main = apply_gate(x_main)
        else:
            x_main = model_instance.blocks(x_main)
            if self.pose_gate_insert_stage.startswith("after_block"):
                print("Warning: Blocks are not in a ModuleList, unable to insert at specific block level. Skipping block-level gating.")
            
        if hasattr(model_instance, 'norm') and model_instance.norm is not None:
            x_main = model_instance.norm(x_main)
            
        x_out = model_instance.forward_head(x_main)
        
        return x_out

    def get_config(self,):
        if self.share_weights:
            data_config = timm.data.resolve_model_data_config(self.model)
        else:
            data_config = timm.data.resolve_model_data_config(self.model1)
        return data_config
    
    
    def set_grad_checkpointing(self, enable=True):
        if self.share_weights:
            self.model.set_grad_checkpointing(enable)
        else:
            self.model1.set_grad_checkpointing(enable)
            self.model2.set_grad_checkpointing(enable)

    def freeze_layers(self, frozen_blocks=10, frozen_stages=[0,0,0,0]):
        pass

    def forward(self, img1=None, img2=None, pose=None, forward_features=False):

        if self.share_weights:
            if img1 is not None and img2 is not None:
                # Apply pose attention only to img1 (UAV)
                image_features1 = self._apply_pose_attention(self.model, img1, pose)
                image_features2 = self.model(img2)
                if forward_features:
                    return image_features1, None, image_features2, None
                return image_features1, image_features2            
            elif img1 is not None:
                # Apply pose attention to img1 (UAV)
                image_features = self._apply_pose_attention(self.model, img1, pose)
                return image_features
            else:
                image_features = self.model(img2)
                return image_features
        else:
            if img1 is not None and img2 is not None:
                # Apply pose attention only to img1 (UAV)
                image_features1 = self._apply_pose_attention(self.model1, img1, pose)
                image_features2 = self.model2(img2)
                if forward_features:
                    return image_features1, None, image_features2, None
                return image_features1, image_features2            
            elif img1 is not None:
                # Apply pose attention to img1 (UAV)
                image_features = self._apply_pose_attention(self.model1, img1, pose)
                return image_features
            else:
                image_features = self.model2(img2)
                return image_features

    def offset_pred(self, img_feature1, img_feature2):
        offset = self.MLP(torch.cat((img_feature1, img_feature2), dim=1))
        return offset


if __name__ == '__main__':
    # model = TimmModel(model_name='timm/vit_large_patch16_384.augreg_in21k_ft_in1k')
    # # model = TimmModel(model_name='timm/vit_base_patch16_224.augreg_in1k')
    # # from timm.models.vision_transformer import vit_base_patch16_224
    # # model = vit_base_patch16_224(img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, num_classes=0)


    # model = DesModel(model_name='timm/resnet101.tv_in1k', img_size=384)
    # model = DesModel(model_name='convnext_base.fb_in22k_ft_in1k_384', img_size=384)
    model = DesModel(model_name='timm/swin_base_patch4_window7_224.ms_in22k_ft_in1k', img_size=384)
    # # model = TimmModel(model_name='vit_base_patch16_rope_reg1_gap_256.sbb_in1k')
    # # model = TimmModel(model_name='timm/vit_medium_patch16_rope_reg1_gap_256.sbb_in1k')
    # # model = TimmModel(model_name='timm/vit_medium_patch16_gap_256.sw_in12k_ft_in1k')
    # # model = TimmModel(model_name='timm/resnet101.tv_in1k') 
    # # img = Image.open(urlopen(
    # # 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
    # # ))
    x = torch.rand((1, 3, 384, 384))
    x = x.cuda()
    model.cuda()
    x = model(x)
    print(x.shape)

    # flops, params = profile(model, inputs=(x,))
    # # print(img.size)
    # # img = transform(img)
    # # print(img.size)

    # # print(model1)
    # print('flops(G)', flops/1e9, 'params(M)', params/1e6)

    # from transformers import CLIPProcessor, CLIPModel
    # model = CLIPModel.from_pretrained("/home/xmuairmud/jyx/clip-vit-base-patch16")
    # vision_model = model.vision_model
    # print(vision_model)

    # dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
    # print(dinov2_vitb14_reg.set_grad_checkpointing(True))

    # from transformers import ViTModel, ViTImageProcessor, AutoModelForImageClassification, AutoConfig
    # config = AutoConfig.from_pretrained('facebook/dino-vitb16')
    # config.image_size = 384
    # model = ViTModel.from_pretrained('facebook/dino-vitb16', config=config, ignore_mismatched_sizes=True)
    # model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=True, img_size=(384, 384))
    # data_config = timm.data.resolve_model_data_config(model)
    # print(data_config)
    # processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16')


    # x = torch.rand((1, 3, 384, 384))
    # inputs = processor(images=x, return_tensors="pt")
    # print(inputs['pixel_values'].shape)
    # outputs = model(**inputs)
    # print(outputs.pooler_output.shape)
    # print(model(x).shape)
    # flops, params = profile(dinov2_vitb14_reg, inputs=(x,))
    # print('flops(G)', flops/1e9, 'params(M)', params/1e6)


