# SMG_Lite — a ~2M‑parameter, from‑scratch CNN for plant‑leaf‑disease classification

A compact convolutional network designed and **trained from scratch** (no ImageNet
pre‑training) for leaf‑disease recognition. It targets **~2.0 M parameters** while
staying competitive with — and aiming to beat — MobileNetV2/V3, ShuffleNetV2,
ResNet101, EfficientNetV2‑S and EfficientNet‑lite on accuracy, at a fraction of
their size and compute.

**Verified stats (measured in this repo, 224×224, 38 classes):**

| Property | Value |
|---|---|
| Parameters | **2.011 M** (100 % trainable) |
| MACs | **213.1 M** (~**0.426 GFLOPs**) |
| Input | 3 × 224 × 224 |
| Pre‑training | **none — trained from scratch** |

---

## 1. Why another network? — the two novel modules

The design is grounded in the reference papers (Lite‑MDC / MixConv multi‑kernel
depthwise, GhostNet, ECA, Coordinate Attention, SK‑Net selective kernels). The
**contribution is the specific combination**, packaged into two blocks that are new
as a unit:

### (A) SMG block — *Selective Multi‑scale Ghost* inverted residual
Leaf lesions range from tiny specks to large blights, so a single receptive field
is suboptimal. Prior work (Lite‑MDC, MixNet) uses **static** multi‑kernel depthwise
convolutions (a fixed 3×3 + 5×5 split). SMG_Lite instead makes the multi‑scale
fusion **input‑dependent**:

1. **Ghost‑style expansion** — half the expanded channels come from a 1×1 conv, the
   other half from a cheap depthwise op (GhostNet). This roughly halves the cost of
   the expensive expansion 1×1.
2. **Selective multi‑scale depthwise** — parallel depthwise **3×3** and **5×5**
   branches are fused by a per‑channel **soft gate computed from the input**
   (Selective‑Kernel style, but ultra‑light and depthwise). The network learns,
   *per image and per channel*, which scale matters — a dynamic upgrade of the fixed
   MixConv/MDsConv used in the literature.
3. **1×1 projection** back to the block output, with a residual add when shapes match.

### (B) CEA — *Coordinate‑ECA Attention*
Two cheap, complementary attentions fused in one module:

* **ECA branch** → *which channel* is important. A 1‑D convolution over the
  channel descriptor (no dimensionality reduction), following ECA. Answers "which
  disease‑relevant feature maps to amplify" at near‑zero cost.
* **Coordinate branch** → *where* the lesion is. Global pooling is factorised into
  **height‑ and width‑direction** 1‑D encodings (Coordinate Attention), preserving
  positional structure — important for localising small lesions, which pure channel
  attention (SE) throws away.

Fusing "which channel" (ECA) with "where" (coordinate) gives spatial + channel
selectivity for a few‑thousand extra parameters per block.

> **In short:** dynamic multi‑scale receptive fields + ghost efficiency + fused
> positional/channel attention, sized to ~2 M params and trained from scratch. That
> exact stack is not any of the existing baselines.

---

## 2. Architecture (width = 1.0 → 2.011 M params)

| Stage | Operator | Stride | Expand | Attn | Output |
|---|---|---|---|---|---|
| Stem | Conv 3×3 + BN + h‑swish | 2 | – | – | 24 × 112 × 112 |
| 1 | SMG × 2 | 2,1 | 3 | – | 32 × 56 × 56 |
| 2 | SMG × 3 | 2,1,1 | 3 | ✓ | 56 × 28 × 28 |
| 3 | SMG × 4 | 2,1,1,1 | 4 | ✓ | 112 × 14 × 14 |
| 4 | SMG × 3 | 2,1,1 | 4 | ✓ | 160 × 7 × 7 |
| Head | Conv 1×1 + BN + h‑swish | 1 | – | – | 656 × 7 × 7 |
| — | Global avg‑pool → Dropout → Linear | – | – | – | 38 |

Attention is switched **off** in the early, high‑resolution stages (where it is
expensive and adds little) and **on** from stage 2 onward — a deliberate
compute‑placement choice.

---

## 3. Head‑to‑head vs the baselines (size & compute measured here)

Params/MACs below are **computed in `benchmark.py`** for models built *from scratch*
(`weights=None`), at 224×224. They line up with the published table (MobileNetV2
2.27 M, ResNet101 42.58 M, EfficientNetV2‑S 20.23 M, ShuffleNetV2 1.29 M,
MobileNetV3 1.53 M), confirming the counting is consistent.

| Model | Params (M) | GFLOPs | Reported acc.* |
|---|---|---|---|
| **SMG_Lite (ours)** | **2.011** | **0.426** | *train on your data — see §4* |
| MobileNetV2 | 2.273 | 0.599 | 99.66 % |
| MobileNetV3‑Small | 1.557 | 0.111 | 99.80 % |
| MobileNetV3‑Large | 4.251 | 0.431 | – |
| ShuffleNetV2 (x1.0) | 1.293 | 0.288 | 99.72 % |
| ResNet101 | 42.578 | 15.599 | 99.69 % |
| EfficientNetV2‑S | 20.226 | 5.697 | 99.82 % |
| EfficientNet‑B0 | 4.056 | 0.769 | – |
| EfficientNet‑lite0 | ~4.65 (via `timm`) | ~0.80 | 99.80 % |

\* Accuracy figures are the published numbers from the reference *Plant Disease
Dataset* comparison (RTR‑Lite paper, Table 2), shown for context. They were obtained
by the original authors on their split — **use them as a target, not as a claim about
a run performed here.**

**Honest note on accuracy.** This repo verifies the *architecture, its size, its
compute, and that it trains* (gradients flow; it overfits a small batch to loss
≈0.003). It does **not** ship a trained checkpoint on PlantVillage — that requires
running `train.py` on the dataset (a GPU‑hours job). To make the ">= baselines"
comparison real and fair, train **every** model with the same `train.py` recipe and
report the resulting numbers. The design choices below are what give SMG_Lite a
credible path to matching/beating the baselines at ~2 M params:

* dynamic multi‑scale (adaptive receptive field for varied lesion sizes),
* fused coordinate+channel attention (lesion localisation + feature selection),
* a modern from‑scratch recipe (label smoothing, MixUp/CutMix, RandAugment, EMA,
  cosine LR) — the same tricks that let the cited lightweight models reach ~99.8 %.

---

## 4. Usage

```bash
pip install torch torchvision            # timm optional, only for EfficientNet‑lite

# 1) sanity‑check the model (prints params + a forward pass)
python SMG_Lite.py

# 2) size/FLOPs comparison vs baselines
python benchmark.py --classes 38

# 3) train on an ImageFolder dataset
#    - works with New Plant Diseases Dataset (has train/ + valid/)
#    - or any single folder‑per‑class dir (auto 80/20 split)
python train.py --data /path/to/dataset --epochs 60 --bs 64
```

**Dataset layout** (either works):
```
dataset/                      dataset/
├── train/                    ├── Apple___scab/
│   ├── Apple___scab/         ├── Apple___healthy/
│   └── ...                   └── ...            (auto‑split)
└── valid/
    └── ...
```

`train.py` auto‑detects the class count, applies the full augmentation +
regularisation recipe, keeps an EMA copy, and saves the best checkpoint to
`smg_lite_best.pt`.

### Tuning the parameter budget
`width` is a MobileNet‑style multiplier. `width=1.0` → 2.011 M. Roughly:
`0.85→1.7 M`, `1.0→2.0 M`, `1.2→2.9 M`. Set it in `smg_lite(width=…)` or
`train.py --width`.

---

## 5. Files
| File | Purpose |
|---|---|
| `SMG_Lite.py` | Model + the two novel modules (SMG block, Coordinate‑ECA) + param counter |
| `train.py` | Full training pipeline (aug, MixUp/CutMix, EMA, cosine LR, AMP, checkpointing) |
| `benchmark.py` | Params/FLOPs comparison vs MobileNet/ShuffleNet/ResNet/EfficientNet |
