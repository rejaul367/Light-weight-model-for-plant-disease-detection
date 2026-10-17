"""
SMG_Lite: a lightweight (~2M-parameter) CNN built FROM SCRATCH for
plant-leaf-disease classification.

Two novel building blocks (their *combination* is the contribution):

  1. SMG block  - Selective Multi-scale Ghost inverted-residual block.
                  * Ghost-style expansion  (cheap feature generation, GhostNet idea)
                  * Selective multi-scale depthwise (parallel 3x3 & 5x5 depthwise
                    fused by an INPUT-DEPENDENT per-channel soft gate -- an
                    SK-Net-style dynamic version of MixConv / MDsConv, so the
                    network chooses the receptive field per image instead of a
                    fixed split).
  2. CEA        - Coordinate-ECA Attention.
                  * ECA branch  -> "which channel"  (1-D conv, no dim-reduction)
                  * Coordinate branch -> "where"     (H- and W-direction pooling,
                    preserves positional cues for lesion localisation)
                  fused at near-zero cost.

Author: built for the plant-disease lightweight-model project.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
def _make_divisible(v, divisor=8):
    """Round channel count to a multiple of `divisor` (keeps it even for Ghost)."""
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)


# --------------------------------------------------------------------------- #
#  Ghost-style expansion  (GhostNet, Han et al. 2020)
# --------------------------------------------------------------------------- #
class GhostExpand(nn.Module):
    """Produce `cout` channels: half by a 1x1 conv, half by a cheap depthwise op."""

    def __init__(self, cin, cout, dw_k=3, act=True):
        super().__init__()
        assert cout % 2 == 0, "GhostExpand needs an even output width"
        init_c = cout // 2
        Act = nn.Hardswish if act else nn.Identity
        self.primary = nn.Sequential(
            nn.Conv2d(cin, init_c, 1, bias=False),
            nn.BatchNorm2d(init_c),
            Act(),
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(init_c, init_c, dw_k, 1, dw_k // 2, groups=init_c, bias=False),
            nn.BatchNorm2d(init_c),
            Act(),
        )

    def forward(self, x):
        y = self.primary(x)
        z = self.cheap(y)
        return torch.cat([y, z], dim=1)


# --------------------------------------------------------------------------- #
#  NOVEL 1 : Selective Multi-scale Depthwise  (dynamic MixConv / SK-style)
# --------------------------------------------------------------------------- #
class SelectiveMultiScaleDW(nn.Module):
    """
    Parallel depthwise 3x3 and 5x5 branches fused by a per-channel soft gate that
    is computed from the input.  Lesions span many scales; the gate lets the
    network emphasise the right receptive field for each image / channel instead
    of using a fixed multi-kernel split.
    """

    def __init__(self, ch, stride=1, reduction=8):
        super().__init__()
        self.dw3 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, stride, 1, groups=ch, bias=False), nn.BatchNorm2d(ch)
        )
        self.dw5 = nn.Sequential(
            nn.Conv2d(ch, ch, 5, stride, 2, groups=ch, bias=False), nn.BatchNorm2d(ch)
        )
        d = max(ch // reduction, 8)
        self.squeeze = nn.Sequential(
            nn.Conv2d(ch, d, 1, bias=False), nn.BatchNorm2d(d), nn.ReLU(inplace=True)
        )
        self.att3 = nn.Conv2d(d, ch, 1)
        self.att5 = nn.Conv2d(d, ch, 1)
        self.act = nn.Hardswish()

    def forward(self, x):
        u3 = self.dw3(x)
        u5 = self.dw5(x)
        u = u3 + u5
        s = u.mean((2, 3), keepdim=True)          # global context
        z = self.squeeze(s)
        w = torch.stack([self.att3(z), self.att5(z)], dim=0)  # (2,B,C,1,1)
        w = torch.softmax(w, dim=0)               # per-channel branch weights
        out = w[0] * u3 + w[1] * u5
        return self.act(out)


# --------------------------------------------------------------------------- #
#  NOVEL 2 : Coordinate-ECA Attention  (ECA channel  +  coordinate spatial)
# --------------------------------------------------------------------------- #
class CoordECA(nn.Module):
    """ECA answers 'which channel', coordinate pooling answers 'where'."""

    def __init__(self, ch, gamma=2, b=1, reduction=16):
        super().__init__()
        # ---- ECA: adaptive 1-D kernel ----
        t = int(abs((math.log2(ch) + b) / gamma))
        k = t if t % 2 else t + 1
        k = max(k, 3)
        self.eca = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)
        # ---- coordinate attention ----
        d = max(ch // reduction, 8)
        self.reduce = nn.Sequential(
            nn.Conv2d(ch, d, 1, bias=False), nn.BatchNorm2d(d), nn.Hardswish()
        )
        self.conv_h = nn.Conv2d(d, ch, 1)
        self.conv_w = nn.Conv2d(d, ch, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        # ---- channel attention (ECA) ----
        y = x.mean((2, 3))                              # (B,C)
        y = self.eca(y.unsqueeze(1)).squeeze(1)         # (B,C)
        x = x * torch.sigmoid(y).view(b, c, 1, 1)
        # ---- positional attention (coordinate) ----
        xh = x.mean(3, keepdim=True)                    # (B,C,H,1)
        xw = x.mean(2, keepdim=True).permute(0, 1, 3, 2)  # (B,C,W,1)
        z = self.reduce(torch.cat([xh, xw], dim=2))     # (B,d,H+W,1)
        zh, zw = torch.split(z, [h, w], dim=2)
        ah = torch.sigmoid(self.conv_h(zh))             # (B,C,H,1)
        aw = torch.sigmoid(self.conv_w(zw.permute(0, 1, 3, 2)))  # (B,C,1,W)
        return x * ah * aw


# --------------------------------------------------------------------------- #
#  SMG block  (inverted residual using the two modules above)
# --------------------------------------------------------------------------- #
class SMGBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, expand=3, use_attn=True):
        super().__init__()
        self.use_res = stride == 1 and cin == cout
        if expand != 1:
            mid = _make_divisible(cin * expand)
            self.expand = GhostExpand(cin, mid, act=True)
        else:
            mid = cin
            self.expand = nn.Identity()
        self.ms = SelectiveMultiScaleDW(mid, stride=stride)
        self.attn = CoordECA(mid) if use_attn else nn.Identity()
        self.project = nn.Sequential(
            nn.Conv2d(mid, cout, 1, bias=False), nn.BatchNorm2d(cout)
        )

    def forward(self, x):
        idn = x
        out = self.project(self.attn(self.ms(self.expand(x))))
        return out + idn if self.use_res else out


# --------------------------------------------------------------------------- #
#  SMG_Lite
# --------------------------------------------------------------------------- #
class SMG_Lite(nn.Module):
    """
    width ~1.0  -> ~2.0 M parameters at 224x224.
    """

    # (out_channels, stride, expand, use_attention)  -- width=1.0 => ~2.0M params
    BASE_CFG = [
        (32, 2, 3, False),   # stage-1  112 -> 56
        (32, 1, 3, False),
        (56, 2, 3, True),    # stage-2  56  -> 28
        (56, 1, 3, True),
        (56, 1, 3, True),
        (112, 2, 4, True),   # stage-3  28  -> 14
        (112, 1, 4, True),
        (112, 1, 4, True),
        (112, 1, 4, True),
        (160, 2, 4, True),   # stage-4  14  -> 7
        (160, 1, 4, True),
        (160, 1, 4, True),
    ]

    def __init__(self, num_classes=38, width=1.0, dropout=0.2, last_channel=656):
        super().__init__()

        def c(ch):
            return _make_divisible(ch * width)

        stem_c = c(24)
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_c, 3, 2, 1, bias=False),
            nn.BatchNorm2d(stem_c),
            nn.Hardswish(),
        )

        blocks, cin = [], stem_c
        for out, s, e, a in self.BASE_CFG:
            oc = c(out)
            blocks.append(SMGBlock(cin, oc, stride=s, expand=e, use_attn=a))
            cin = oc
        self.blocks = nn.Sequential(*blocks)

        last_c = c(last_channel)
        self.head_conv = nn.Sequential(
            nn.Conv2d(cin, last_c, 1, bias=False),
            nn.BatchNorm2d(last_c),
            nn.Hardswish(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(last_c, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head_conv(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def smg_lite(num_classes=38, width=1.0, **kw):
    return SMG_Lite(num_classes=num_classes, width=width, **kw)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    m = smg_lite(num_classes=38, width=1.0)
    tot, tr = count_parameters(m)
    y = m(torch.randn(2, 3, 224, 224))
    print(f"[default width=1.0]  params={tot/1e6:.3f}M  trainable={tr/1e6:.3f}M  out={tuple(y.shape)}")
    # per-section parameter breakdown
    for name, mod in [("stem", m.stem), ("blocks", m.blocks),
                      ("head_conv", m.head_conv), ("classifier", m.classifier)]:
        p = sum(x.numel() for x in mod.parameters())
        print(f"    {name:11s}: {p/1e6:.3f}M")
