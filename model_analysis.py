import torch
from ultralytics.models.yolo import YOLO
from thop import profile, clever_format

# Config
model_path = r"...\best.pt"
input_shape = (3, 640, 640)

# Load model
model = YOLO(model_path)
model.model.train()  # Enable train mode to check gradients
print(f"Model loaded successfully: {model_path}")

# Layer-wise parameter statistics
print("\n" + "-" * 50)
print("Layer-wise Parameter Statistics")
print("-" * 50)
total_params = 0
for idx, (name, layer) in enumerate(model.model.named_modules()):
    # Skip layers without parameters
    layer_params_list = list(layer.parameters())
    if not layer_params_list:
        continue

    # Calculate current layer's parameters
    layer_params = sum(p.numel() for p in layer_params_list)
    total_params += layer_params
    print(f"Layer {idx:3d} | {name:<40} | Params: {layer_params / 1e6:.4f} M")

print("-" * 50)
print(f"Total Params: {total_params / 1e6:.2f} M ({total_params:,} params)")
print("-" * 50)

# Layer-wise gradient check
print("\n" + "-" * 50)
print("Layer-wise Gradient Status")
print("-" * 50)
grad_exist_flag = False
for idx, (name, layer) in enumerate(model.model.named_modules()):
    layer_params_list = list(layer.parameters())
    if not layer_params_list:
        continue

    # Check if gradient exists and is non-zero
    has_grad = False
    for p in layer_params_list:
        if p.grad is not None and torch.sum(p.grad) != 0:
            has_grad = True
            break
    if has_grad:
        grad_exist_flag = True
        print(f"Layer {idx:3d} | {name:<40} | Gradient exists")
    else:
        print(f"Layer {idx:3d} | {name:<40} | No gradient/Zero gradient")

if not grad_exist_flag:
    print("\nNote: No valid gradients found (Model not trained on sample data yet)")
print("-" * 50)

# GFLOPs calculation
print("\n" + "-" * 50)
print("GFLOPs Calculation")
print("-" * 50)
# Create dummy input
dummy_input = torch.randn(1, *input_shape)
# Calculate flops and params
flops, params = profile(model.model, inputs=(dummy_input,), verbose=False)
# Format units
flops_fmt, params_fmt = clever_format([flops, params], "%.2f")
gflops = flops / 1e9

print(f"Input Shape: {input_shape}")
print(f"Total FLOPs: {flops_fmt} ({gflops:.2f} GFLOPs)")
print(f"Params (thop): {params_fmt}")
print("-" * 50)

# Simple summary
print(f"\nSummary: Total Params {total_params / 1e6:.2f} M, GFLOPs {gflops:.2f} G")

print("Analysis completed!")
