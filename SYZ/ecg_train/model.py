"""model -- 1-D architectures for 12-lead ECG classification.

Every model takes two inputs and is built to export cleanly to ONNX:

    x : (B, 12, T)   preprocessed signal from ecg_preprocess.preprocess_signal
    f : (B, 37)      hand features from ecg_preprocess.extract_features

The feature scaler lives *inside* the module as buffers, so the exported ONNX
graph is self-contained: the package never has to carry a separate scaler file
that could drift out of sync with the weights.

Presets are selected with ``--preset``. Widths follow the one measured signal
we have (base 32 -> 48 gained +0.018 on fold 0), so w64 and w80 extend that
line rather than guessing in a new direction.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_LEADS = 12
N_FEATURES = 37
N_CLASSES = 5


PRESETS = {
    # --- ResNet family: width sweep on a fixed depth (FAZ 2) ---
    "r18":    dict(arch="resnet", base=32, blocks=(2, 2, 2, 2),
                   mults=(1, 2, 4, 8), k=7,  stem_k=15),
    "r34":    dict(arch="resnet", base=32, blocks=(3, 4, 6, 3),
                   mults=(1, 2, 4, 8), k=7,  stem_k=15),
    "r18k11": dict(arch="resnet", base=32, blocks=(2, 2, 2, 2),
                   mults=(1, 2, 4, 8), k=11, stem_k=21),
    "wide":   dict(arch="resnet", base=48, blocks=(2, 2, 2, 2),
                   mults=(1, 2, 4, 8), k=7,  stem_k=15),
    "w64":    dict(arch="resnet", base=64, blocks=(2, 2, 2, 2),
                   mults=(1, 2, 4, 8), k=7,  stem_k=15),
    "w80":    dict(arch="resnet", base=80, blocks=(2, 2, 2, 2),
                   mults=(1, 2, 4, 8), k=7,  stem_k=15),
    # --- genuinely different inductive biases (FAZ 5) ---
    "inception": dict(arch="inception", base=32, blocks=(2, 2, 2, 2),
                      mults=(1, 2, 4, 8), kernels=(11, 21, 41)),
    "hybrid":    dict(arch="hybrid", base=32, blocks=(2, 2, 2),
                      mults=(1, 2, 4), k=7, stem_k=15,
                      d_model=192, heads=4, layers=2, ff=384),
}


def _norm(ch):
    return nn.BatchNorm1d(ch)


class SEBlock(nn.Module):
    """Squeeze-and-excitation over channels; cheap and reliably helps here."""

    def __init__(self, ch, reduction=8):
        super().__init__()
        hidden = max(ch // reduction, 4)
        self.fc1 = nn.Conv1d(ch, hidden, 1)
        self.fc2 = nn.Conv1d(hidden, ch, 1)

    def forward(self, x):
        w = x.mean(dim=2, keepdim=True)
        w = torch.relu(self.fc1(w))
        return x * torch.sigmoid(self.fc2(w))


class BasicBlock1d(nn.Module):
    """Pre-activation residual block with optional SE and channel dropout."""

    def __init__(self, in_ch, out_ch, k=7, stride=1, dropout=0.0, se=True):
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, stride=stride, padding=pad,
                               bias=False)
        self.bn1 = _norm(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=pad, bias=False)
        self.bn2 = _norm(out_ch)
        self.se = SEBlock(out_ch) if se else None
        self.drop = nn.Dropout(dropout) if dropout > 0 else None

        if stride != 1 or in_ch != out_ch:
            self.short = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                _norm(out_ch))
        else:
            self.short = None

    def forward(self, x):
        identity = x if self.short is None else self.short(x)
        y = torch.relu(self.bn1(self.conv1(x)))
        if self.drop is not None:
            y = self.drop(y)
        y = self.bn2(self.conv2(y))
        if self.se is not None:
            y = self.se(y)
        return torch.relu(y + identity)


class InceptionBlock1d(nn.Module):
    """Parallel kernels: the network picks its own time scale per stage.

    A ResNet with k=7 has one receptive-field growth rate. Running 11 / 21 / 41
    side by side is a different inductive bias, which is the point of FAZ 5 --
    an ensemble only gains from a model that makes *different* mistakes.
    """

    def __init__(self, in_ch, out_ch, kernels=(11, 21, 41), stride=1,
                 dropout=0.0):
        super().__init__()
        n_branch = len(kernels) + 1
        branch_ch = max(out_ch // n_branch, 8)
        self.bottleneck = nn.Conv1d(in_ch, branch_ch, 1, bias=False) \
            if in_ch > branch_ch else None
        mid = branch_ch if self.bottleneck is not None else in_ch

        self.convs = nn.ModuleList([
            nn.Conv1d(mid, branch_ch, k, stride=stride, padding=k // 2,
                      bias=False) for k in kernels])
        self.pool_conv = nn.Conv1d(in_ch, branch_ch, 1, bias=False)
        self.stride = stride

        total = branch_ch * n_branch
        self.bn = _norm(total)
        self.project = nn.Conv1d(total, out_ch, 1, bias=False) \
            if total != out_ch else None
        self.bn_out = _norm(out_ch) if self.project is not None else None
        self.drop = nn.Dropout(dropout) if dropout > 0 else None

        if stride != 1 or in_ch != out_ch:
            self.short = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                _norm(out_ch))
        else:
            self.short = None

    def forward(self, x):
        identity = x if self.short is None else self.short(x)
        mid = x if self.bottleneck is None else self.bottleneck(x)
        branches = [conv(mid) for conv in self.convs]

        pooled = F.max_pool1d(x, 3, stride=self.stride, padding=1)
        branches.append(self.pool_conv(pooled))

        length = min(b.shape[-1] for b in branches)
        y = torch.cat([b[..., :length] for b in branches], dim=1)
        y = torch.relu(self.bn(y))
        if self.project is not None:
            y = self.bn_out(self.project(y))
        if self.drop is not None:
            y = self.drop(y)
        return torch.relu(y + identity[..., :y.shape[-1]])


class TransformerBlock(nn.Module):
    """Post-norm self-attention block, written out so ONNX export is exact."""

    def __init__(self, d_model, heads, ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout,
                                          batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm1(x)
        attended, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(attended)
        return x + self.drop(self.ff(self.norm2(x)))


# --------------------------------------------------------------------------
# backbones
# --------------------------------------------------------------------------

class ResNetBackbone(nn.Module):
    def __init__(self, cfg, in_ch=N_LEADS, dropout=0.0):
        super().__init__()
        base = cfg["base"]
        stem_k = cfg.get("stem_k", 15)
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base, stem_k, stride=2, padding=stem_k // 2,
                      bias=False),
            _norm(base), nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1))

        layers, ch = [], base
        for stage, (n_blocks, mult) in enumerate(zip(cfg["blocks"], cfg["mults"])):
            out_ch = base * mult
            for b in range(n_blocks):
                layers.append(BasicBlock1d(
                    ch, out_ch, k=cfg.get("k", 7),
                    stride=2 if (b == 0 and stage > 0) else 1,
                    dropout=dropout, se=cfg.get("se", True)))
                ch = out_ch
        self.layers = nn.Sequential(*layers)
        self.out_channels = ch

    def forward(self, x):
        return self.layers(self.stem(x))


class InceptionBackbone(nn.Module):
    def __init__(self, cfg, in_ch=N_LEADS, dropout=0.0):
        super().__init__()
        base = cfg["base"]
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base, 15, stride=2, padding=7, bias=False),
            _norm(base), nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1))

        layers, ch = [], base
        for stage, (n_blocks, mult) in enumerate(zip(cfg["blocks"], cfg["mults"])):
            out_ch = base * mult
            for b in range(n_blocks):
                layers.append(InceptionBlock1d(
                    ch, out_ch, kernels=cfg.get("kernels", (11, 21, 41)),
                    stride=2 if (b == 0 and stage > 0) else 1, dropout=dropout))
                ch = out_ch
        self.layers = nn.Sequential(*layers)
        self.out_channels = ch

    def forward(self, x):
        return self.layers(self.stem(x))


class HybridBackbone(nn.Module):
    """CNN encoder, then self-attention over the downsampled time axis.

    The convolutions handle local morphology; attention lets any beat see any
    other beat, which is the natural way to judge rhythm regularity over a 10 s
    strip. Sinusoidal position codes are a fixed buffer, so nothing about the
    sequence length is learned and export stays simple.
    """

    def __init__(self, cfg, in_ch=N_LEADS, dropout=0.1, max_len=2048):
        super().__init__()
        base = cfg["base"]
        stem_k = cfg.get("stem_k", 15)
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base, stem_k, stride=2, padding=stem_k // 2,
                      bias=False),
            _norm(base), nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1))

        layers, ch = [], base
        for stage, (n_blocks, mult) in enumerate(zip(cfg["blocks"], cfg["mults"])):
            out_ch = base * mult
            for b in range(n_blocks):
                layers.append(BasicBlock1d(
                    ch, out_ch, k=cfg.get("k", 7),
                    stride=2 if (b == 0 and stage > 0) else 1,
                    dropout=dropout * 0.5, se=True))
                ch = out_ch
        self.conv = nn.Sequential(*layers)

        d_model = cfg.get("d_model", 192)
        self.project = nn.Conv1d(ch, d_model, 1)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, cfg.get("heads", 4), cfg.get("ff", 384),
                             dropout)
            for _ in range(cfg.get("layers", 2))])
        self.norm = nn.LayerNorm(d_model)
        self.out_channels = d_model

        pos = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pos[:, 0::2] = torch.sin(position * div)
        pos[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pos_encoding", pos.unsqueeze(0), persistent=False)

    def forward(self, x):
        h = self.project(self.conv(self.stem(x)))       # (B, d, T')
        h = h.transpose(1, 2)                           # (B, T', d)
        h = h + self.pos_encoding[:, :h.shape[1], :]
        for block in self.blocks:
            h = block(h)
        return self.norm(h).transpose(1, 2)             # (B, d, T')


# --------------------------------------------------------------------------
# full model
# --------------------------------------------------------------------------

class ECGNet(nn.Module):
    """Backbone over the signal + an MLP over the 37 features, then a head."""

    def __init__(self, preset="r18", dropout=0.2, n_classes=N_CLASSES,
                 n_features=N_FEATURES, in_ch=N_LEADS, use_features=True):
        super().__init__()
        if preset not in PRESETS:
            raise KeyError("unknown preset %r; known: %s"
                           % (preset, ", ".join(sorted(PRESETS))))
        cfg = dict(PRESETS[preset])
        self.preset = preset
        self.cfg = cfg
        self.use_features = use_features

        arch = cfg.get("arch", "resnet")
        if arch == "resnet":
            self.backbone = ResNetBackbone(cfg, in_ch, dropout * 0.5)
        elif arch == "inception":
            self.backbone = InceptionBackbone(cfg, in_ch, dropout * 0.5)
        elif arch == "hybrid":
            self.backbone = HybridBackbone(cfg, in_ch, dropout)
        else:
            raise KeyError("unknown arch %r" % arch)

        pooled = self.backbone.out_channels * 2      # avg ++ max

        # Feature standardisation baked in: filled by load_feature_scaler()
        # from the training fold only, then frozen into the ONNX graph.
        self.register_buffer("feat_mean", torch.zeros(n_features))
        self.register_buffer("feat_std", torch.ones(n_features))

        feat_dim = 0
        if use_features:
            feat_dim = 64
            self.feat_branch = nn.Sequential(
                nn.Linear(n_features, 128), nn.BatchNorm1d(128),
                nn.ReLU(inplace=True), nn.Dropout(dropout),
                nn.Linear(128, feat_dim), nn.BatchNorm1d(feat_dim),
                nn.ReLU(inplace=True))
        else:
            self.feat_branch = None

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(pooled + feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, n_classes))

    def set_feature_scaler(self, mean, std):
        """Freeze the fold's feature statistics into the module."""
        mean = torch.as_tensor(mean, dtype=torch.float32).reshape(-1)
        std = torch.as_tensor(std, dtype=torch.float32).reshape(-1)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        self.feat_mean.copy_(mean)
        self.feat_std.copy_(std)

    def forward(self, x, f=None):
        h = self.backbone(x)
        pooled = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)

        if self.feat_branch is not None:
            if f is None:
                f = torch.zeros(x.shape[0], self.feat_mean.shape[0],
                                dtype=x.dtype, device=x.device)
            f = (f - self.feat_mean) / self.feat_std
            f = torch.clamp(f, -10.0, 10.0)
            pooled = torch.cat([pooled, self.feat_branch(f)], dim=1)

        return self.head(pooled)


def build_model(preset="r18", dropout=0.2, use_features=True, **kwargs):
    return ECGNet(preset=preset, dropout=dropout, use_features=use_features,
                  **kwargs)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def describe(preset, input_len=1500):
    """Parameter count and output shape for one preset."""
    model = build_model(preset).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, N_LEADS, input_len), torch.zeros(2, N_FEATURES))
    return {"preset": preset, "params": count_parameters(model),
            "out_shape": tuple(out.shape),
            "arch": PRESETS[preset].get("arch", "resnet")}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="list presets with parameter counts")
    ap.add_argument("--len", type=int, default=1500)
    args = ap.parse_args()

    print("%-10s %-10s %12s  %s" % ("preset", "arch", "params", "out"))
    for name in PRESETS:
        info = describe(name, args.len)
        print("%-10s %-10s %12s  %s"
              % (name, info["arch"], "{:,}".format(info["params"]),
                 info["out_shape"]))
