"""
benchmark.py -- compare SMG_Lite against the standard baselines on
parameter count and FLOPs (all built FROM SCRATCH, i.e. weights=None).

This reproduces the size/compute columns of the comparison table.  To fill in
the accuracy column, train each backbone on YOUR dataset with the same recipe
(train.py) -- SMG_Lite uses identical augmentation/optimiser, so it is a fair
head-to-head.

  python benchmark.py --classes 38
"""

import argparse
import torch
import torch.nn as nn
from torchvision import models

from SMG_Lite import smg_lite, count_parameters


def macs_of(model, size=224):
    """Count multiply-accumulates with forward hooks (Conv2d/Conv1d/Linear)."""
    total = {"n": 0}

    def hook(mod, inp, out):
        if isinstance(mod, nn.Conv2d):
            oc = mod.out_channels
            ic = mod.in_channels // mod.groups
            kh, kw = mod.kernel_size
            oh, ow = out.shape[2], out.shape[3]
            total["n"] += oc * ic * kh * kw * oh * ow
        elif isinstance(mod, nn.Conv1d):
            oc = mod.out_channels
            ic = mod.in_channels // mod.groups
            k = mod.kernel_size[0]
            total["n"] += oc * ic * k * out.shape[-1]
        elif isinstance(mod, nn.Linear):
            total["n"] += mod.in_features * mod.out_features

    hs = [m.register_forward_hook(hook)
          for m in model.modules()
          if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear))]
    model.eval()
    with torch.no_grad():
        model(torch.randn(1, 3, size, size))
    for h in hs:
        h.remove()
    return total["n"]


def build_baselines(num_classes):
    """All from scratch (weights=None); classifier resized to num_classes."""
    reg = {}

    def add(name, fn):
        try:
            reg[name] = fn()
        except Exception as e:
            print(f"  (skip {name}: {e})")

    def mobilenet_v2():
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m

    def mobilenet_v3_small():
        m = models.mobilenet_v3_small(weights=None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
        return m

    def mobilenet_v3_large():
        m = models.mobilenet_v3_large(weights=None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
        return m

    def shufflenet_v2():
        m = models.shufflenet_v2_x1_0(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    def resnet101():
        m = models.resnet101(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    def efficientnet_v2_s():
        m = models.efficientnet_v2_s(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m

    def efficientnet_b0():
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m

    add("MobileNetV2", mobilenet_v2)
    add("MobileNetV3-Small", mobilenet_v3_small)
    add("MobileNetV3-Large", mobilenet_v3_large)
    add("ShuffleNetV2 (x1.0)", shufflenet_v2)
    add("ResNet101", resnet101)
    add("EfficientNetV2-S", efficientnet_v2_s)
    add("EfficientNet-B0", efficientnet_b0)

    # EfficientNet-lite lives in timm, not torchvision
    try:
        import timm
        reg["EfficientNet-lite0"] = timm.create_model(
            "efficientnet_lite0", pretrained=False, num_classes=num_classes)
    except Exception:
        print("  (EfficientNet-lite0 needs `pip install timm` -- skipped)")

    return reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", type=int, default=38)
    ap.add_argument("--img", type=int, default=224)
    args = ap.parse_args()

    rows = []
    ph = smg_lite(num_classes=args.classes, width=1.0)
    p, _ = count_parameters(ph)
    rows.append(("SMG_Lite (ours)", p, macs_of(ph, args.img)))

    for name, m in build_baselines(args.classes).items():
        p = sum(x.numel() for x in m.parameters())
        rows.append((name, p, macs_of(m, args.img)))

    print(f"\n{'Model':26s} {'Params(M)':>10s} {'MACs(M)':>10s} {'GFLOPs':>9s}")
    print("-" * 58)
    for name, p, mac in rows:
        print(f"{name:26s} {p/1e6:10.3f} {mac/1e6:10.1f} {2*mac/1e9:9.3f}")


if __name__ == "__main__":
    main()
