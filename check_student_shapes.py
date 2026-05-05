import torch
import timm
import models.ghostnetv3_small

model = timm.create_model("ghostnetv3_small", width=1.0, num_classes=10)
x = torch.randn(1, 3, 32, 32)

features = {}
def get_hook(name):
    def hook(m, i, o):
        features[name] = o.shape
    return hook

# Register hooks on top-level blocks
for name, m in model.named_modules():
    if "blocks." in name and len(name.split('.')) == 2:
        m.register_forward_hook(get_hook(name))

model(x)
print("=== GhostNetV3_small (width 1.0) Layers ===")
for name in sorted(features.keys(), key=lambda x: int(x.split('.')[1])):
    print(f"{name}: {features[name]}")
