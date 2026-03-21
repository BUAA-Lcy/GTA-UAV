import torch
import timm
model = timm.create_model('resnet50', pretrained=False, num_classes=0)
x = torch.rand(1, 3, 224, 224)
feat = model.forward_features(x)
out = model.forward_head(feat)
print('resnet50 out:', out.shape)

model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
feat = model.forward_features(x)
out = model.forward_head(feat)
print('vit out:', out.shape)

model = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=0)
feat = model.forward_features(x)
out = model.forward_head(feat)
print('swin out:', out.shape)
