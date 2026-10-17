"""
train.py  --  training pipeline for SMG_Lite on plant-leaf-disease datasets.

Works with any ImageFolder-style dataset, e.g.:
  * PlantVillage / "New Plant Diseases Dataset" (Kaggle) with train/ + valid/ splits
  * or a single folder-per-class directory (auto 80/20 split)

Recipe (chosen to match the SOTA lightweight-plant-disease literature):
  RandomResizedCrop + flips + rotation + colour-jitter + RandAugment + RandomErasing,
  label smoothing 0.1, MixUp/CutMix, AdamW, cosine LR with warm-up, weight EMA, AMP.

Example
-------
  python train.py --data /path/to/New_Plant_Diseases_Dataset --epochs 60 --bs 64
"""

import argparse, os, math, time, copy, random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import RandAugment

from SMG_Lite import smg_lite, count_parameters


# --------------------------------------------------------------------------- #
#  data
# --------------------------------------------------------------------------- #
def build_loaders(root, img_size=224, bs=64, workers=4, val_split=0.2):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(0.25, 0.25, 0.25, 0.05),
        RandAugment(num_ops=2, magnitude=7),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # case 1: explicit train/ + valid|val/ subfolders
    sub = {d.lower(): d for d in os.listdir(root)
           if os.path.isdir(os.path.join(root, d))}
    if "train" in sub and ("valid" in sub or "val" in sub):
        vkey = "valid" if "valid" in sub else "val"
        train_ds = datasets.ImageFolder(os.path.join(root, sub["train"]), train_tf)
        val_ds = datasets.ImageFolder(os.path.join(root, sub[vkey]), eval_tf)
        classes = train_ds.classes
    else:  # case 2: single folder-per-class -> random split
        full = datasets.ImageFolder(root, train_tf)
        n_val = int(len(full) * val_split)
        n_tr = len(full) - n_val
        g = torch.Generator().manual_seed(42)
        train_ds, val_ds = random_split(full, [n_tr, n_val], generator=g)
        val_ds.dataset = copy.copy(full)
        val_ds.dataset.transform = eval_tf  # eval transform for the val split
        classes = full.classes

    train_ld = DataLoader(train_ds, bs, shuffle=True, num_workers=workers,
                          pin_memory=True, drop_last=True)
    val_ld = DataLoader(val_ds, bs, shuffle=False, num_workers=workers,
                        pin_memory=True)
    return train_ld, val_ld, classes


# --------------------------------------------------------------------------- #
#  MixUp / CutMix
# --------------------------------------------------------------------------- #
def rand_bbox(H, W, lam):
    r = math.sqrt(1.0 - lam)
    cw, ch = int(W * r), int(H * r)
    cx, cy = random.randint(0, W), random.randint(0, H)
    x1, y1 = max(cx - cw // 2, 0), max(cy - ch // 2, 0)
    x2, y2 = min(cx + cw // 2, W), min(cy + ch // 2, H)
    return x1, y1, x2, y2


def mix_batch(x, y, alpha=0.2, cutmix_prob=0.5):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    idx = torch.randperm(x.size(0), device=x.device)
    y2 = y[idx]
    if random.random() < cutmix_prob:  # CutMix
        x1, y1, x2, y2b = rand_bbox(x.size(2), x.size(3), lam)
        x[:, :, y1:y2b, x1:x2] = x[idx, :, y1:y2b, x1:x2]
        lam = 1 - ((x2 - x1) * (y2b - y1) / (x.size(2) * x.size(3)))
    else:                              # MixUp
        x = lam * x + (1 - lam) * x[idx]
    return x, y, y2, lam


# --------------------------------------------------------------------------- #
#  weight EMA
# --------------------------------------------------------------------------- #
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for es, ms in zip(self.shadow.state_dict().values(),
                          model.state_dict().values()):
            if es.dtype.is_floating_point:
                es.mul_(self.decay).add_(ms, alpha=1 - self.decay)
            else:
                es.copy_(ms)


# --------------------------------------------------------------------------- #
#  eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / max(total, 1)


# --------------------------------------------------------------------------- #
#  train
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset root (ImageFolder)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=5e-2)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--mixup", type=float, default=0.2)
    ap.add_argument("--smooth", type=float, default=0.1)
    ap.add_argument("--out", default="smg_lite_best.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ld, val_ld, classes = build_loaders(
        args.data, args.img, args.bs, args.workers)
    num_classes = len(classes)
    print(f"device={device}  classes={num_classes}  "
          f"train_batches={len(train_ld)}  val_batches={len(val_ld)}")

    model = smg_lite(num_classes=num_classes, width=args.width).to(device)
    tot, _ = count_parameters(model)
    print(f"SMG_Lite params: {tot/1e6:.3f} M")

    crit = nn.CrossEntropyLoss(label_smoothing=args.smooth)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    steps = len(train_ld)
    warm = args.warmup * steps
    total_steps = args.epochs * steps

    def lr_at(step):  # linear warm-up then cosine decay
        if step < warm:
            return step / max(warm, 1)
        p = (step - warm) / max(total_steps - warm, 1)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))
    ema = EMA(model, decay=0.999)

    best = 0.0
    for ep in range(args.epochs):
        model.train()
        t0, run = time.time(), 0.0
        for x, y in train_ld:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x, ya, yb, lam = mix_batch(x, y, alpha=args.mixup)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.split(":")[0],
                                    enabled=(device == "cuda")):
                out = model(x)
                loss = lam * crit(out, ya) + (1 - lam) * crit(out, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model)
            run += loss.item()

        acc = evaluate(model, val_ld, device)
        acc_ema = evaluate(ema.shadow, val_ld, device)
        best_now = max(acc, acc_ema)
        if best_now > best:
            best = best_now
            save_model = ema.shadow if acc_ema >= acc else model
            torch.save({"model": save_model.state_dict(),
                        "classes": classes,
                        "acc": best,
                        "width": args.width}, args.out)
        print(f"ep {ep+1:3d}/{args.epochs}  loss {run/steps:.3f}  "
              f"val {acc:5.2f}  val_ema {acc_ema:5.2f}  best {best:5.2f}  "
              f"lr {sched.get_last_lr()[0]:.1e}  {time.time()-t0:5.1f}s")

    print(f"\nDONE. best val acc = {best:.2f}%  saved -> {args.out}")


if __name__ == "__main__":
    main()
