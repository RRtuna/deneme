"""model_diverse -- ResNet'ten GERCEKTEN farkli iki mimari (FAZ 5).

Mevcut ensemble'in bes ailesi de ResNet varyanti. 30->20 budama denemesi bunun
bedelini olctu: 10 model cikarildi, 750 tahminin **hicbiri** degismedi. Bes
ResNet, bir ResNet kadar bilgi tasiyor. Ensemble'in kazanabilecegi tek yer,
farkli hatalar yapan bir mimari.

Bu dosya kendi kendine yeter: `model.py`'ye dokunmadan da kullanilabilir.
Girdi/cikti sozlesmesi mevcut modelle ayni:

    forward(x, f) ->  x: (B, 12, T) sinyal
                      f: (B, n_features) el-yapimi ozellikler
                      cikti: (B, 5) logit

Iki mimari:

  inception  Paralel 11/21/41 cekirdek. ResNet'in tek bir alici-alan buyume
             hizi vardir; uc farkli zaman olcegini yan yana kosturmak farkli
             bir tumevarim onyargisidir. AFL'nin ince F dalgasi ile QRS'in
             genis morfolojisi ayni katmanda gorulur.

  hybrid     CNN kodlayici + 2 katmanli self-attention. Konvolusyon yerel
             morfolojiyi cozer; dikkat, 10 saniyedeki herhangi iki vurusun
             birbirini gormesini saglar -- ritim duzenliligini yargilamanin
             dogal yolu budur ve AFIB/AFL ayrimi tam olarak bir duzenlilik
             sorusudur.

Ozellik dali `n_features` ile ayarlanir: 37 (mevcut) ya da QRST ozellikleri
eklenirse 61. Ozellik olcekleyici modulun ICINE gomulur (buffer), boylece
disa aktarilan ONNX grafigi kendi kendine yeter.

model.py'ye baglamak icin (3 satir)
-----------------------------------
    from model_diverse import PRESETS_DIVERSE, build_diverse
    PRESETS.update(PRESETS_DIVERSE)                    # PRESETS tanimindan SONRA

ve mevcut `build_model` fonksiyonunun basina:

    if preset in PRESETS_DIVERSE:
        return build_diverse(preset, dropout=dropout, use_features=use_features)

Boylece `train.py --preset inception` dogrudan calisir.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_LEADS = 12
N_CLASSES = 5

PRESETS_DIVERSE = {
    "inception": dict(arch="inception", base=32, blocks=(2, 2, 2, 2),
                      mults=(1, 2, 4, 8), kernels=(11, 21, 41)),
    "inception_w": dict(arch="inception", base=48, blocks=(2, 2, 2, 2),
                        mults=(1, 2, 4, 8), kernels=(11, 21, 41)),
    "hybrid": dict(arch="hybrid", base=32, blocks=(2, 2, 2),
                   mults=(1, 2, 4), k=7, stem_k=15,
                   d_model=192, heads=4, layers=2, ff=384),
}


def _bn(ch):
    return nn.BatchNorm1d(ch)


class SEBlock(nn.Module):
    """Kanal dikkati -- ucuz ve bu problemde guvenilir sekilde faydali."""

    def __init__(self, ch, reduction=8):
        super().__init__()
        hidden = max(ch // reduction, 4)
        self.fc1 = nn.Conv1d(ch, hidden, 1)
        self.fc2 = nn.Conv1d(hidden, ch, 1)

    def forward(self, x):
        w = x.mean(dim=2, keepdim=True)
        return x * torch.sigmoid(self.fc2(torch.relu(self.fc1(w))))


class BasicBlock1d(nn.Module):
    def __init__(self, in_ch, out_ch, k=7, stride=1, dropout=0.0, se=True):
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, stride=stride, padding=pad, bias=False)
        self.bn1 = _bn(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=pad, bias=False)
        self.bn2 = _bn(out_ch)
        self.se = SEBlock(out_ch) if se else None
        self.drop = nn.Dropout(dropout) if dropout > 0 else None
        self.short = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False), _bn(out_ch)
        ) if (stride != 1 or in_ch != out_ch) else None

    def forward(self, x):
        idt = x if self.short is None else self.short(x)
        y = torch.relu(self.bn1(self.conv1(x)))
        if self.drop is not None:
            y = self.drop(y)
        y = self.bn2(self.conv2(y))
        if self.se is not None:
            y = self.se(y)
        return torch.relu(y + idt)


class InceptionBlock1d(nn.Module):
    """Paralel cekirdekler: ag kendi zaman olcegini her katmanda kendi secer."""

    def __init__(self, in_ch, out_ch, kernels=(11, 21, 41), stride=1, dropout=0.0):
        super().__init__()
        n_branch = len(kernels) + 1
        branch_ch = max(out_ch // n_branch, 8)
        self.bottleneck = nn.Conv1d(in_ch, branch_ch, 1, bias=False) \
            if in_ch > branch_ch else None
        mid = branch_ch if self.bottleneck is not None else in_ch

        self.convs = nn.ModuleList([
            nn.Conv1d(mid, branch_ch, k, stride=stride, padding=k // 2, bias=False)
            for k in kernels])
        self.pool_conv = nn.Conv1d(in_ch, branch_ch, 1, bias=False)
        self.stride = stride

        total = branch_ch * n_branch
        self.bn = _bn(total)
        self.project = nn.Conv1d(total, out_ch, 1, bias=False) if total != out_ch else None
        self.bn_out = _bn(out_ch) if self.project is not None else None
        self.drop = nn.Dropout(dropout) if dropout > 0 else None
        self.short = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False), _bn(out_ch)
        ) if (stride != 1 or in_ch != out_ch) else None

    def forward(self, x):
        idt = x if self.short is None else self.short(x)
        mid = x if self.bottleneck is None else self.bottleneck(x)
        branches = [c(mid) for c in self.convs]
        branches.append(self.pool_conv(F.max_pool1d(x, 3, stride=self.stride, padding=1)))
        length = min(b.shape[-1] for b in branches)
        y = torch.relu(self.bn(torch.cat([b[..., :length] for b in branches], dim=1)))
        if self.project is not None:
            y = self.bn_out(self.project(y))
        if self.drop is not None:
            y = self.drop(y)
        return torch.relu(y + idt[..., :y.shape[-1]])


class TransformerBlock(nn.Module):
    """Pre-norm self-attention. ONNX disa aktariminin kesin olmasi icin acikca yazildi."""

    def __init__(self, d_model, heads, ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout,
                                          batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(ff, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(a)
        return x + self.drop(self.ff(self.norm2(x)))


class InceptionBackbone(nn.Module):
    def __init__(self, cfg, in_ch=N_LEADS, dropout=0.0):
        super().__init__()
        base = cfg["base"]
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base, 15, stride=2, padding=7, bias=False),
            _bn(base), nn.ReLU(inplace=True), nn.MaxPool1d(3, stride=2, padding=1))
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
    def __init__(self, cfg, in_ch=N_LEADS, dropout=0.1, max_len=2048):
        super().__init__()
        base, stem_k = cfg["base"], cfg.get("stem_k", 15)
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base, stem_k, stride=2, padding=stem_k // 2, bias=False),
            _bn(base), nn.ReLU(inplace=True), nn.MaxPool1d(3, stride=2, padding=1))
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
            TransformerBlock(d_model, cfg.get("heads", 4), cfg.get("ff", 384), dropout)
            for _ in range(cfg.get("layers", 2))])
        self.norm = nn.LayerNorm(d_model)
        self.out_channels = d_model

        # Sabit sinuzoidal konum kodu: uzunluk hakkinda hicbir sey ogrenilmez,
        # disa aktarim basit kalir.
        pos = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos[:, 0::2] = torch.sin(position * div)
        pos[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pos_encoding", pos.unsqueeze(0), persistent=False)

    def forward(self, x):
        h = self.project(self.conv(self.stem(x))).transpose(1, 2)   # (B, T', d)
        h = h + self.pos_encoding[:, :h.shape[1], :]
        for blk in self.blocks:
            h = blk(h)
        return self.norm(h).transpose(1, 2)


class DiverseNet(nn.Module):
    def __init__(self, preset, dropout=0.2, n_features=37, n_classes=N_CLASSES,
                 in_ch=N_LEADS, use_features=True):
        super().__init__()
        if preset not in PRESETS_DIVERSE:
            raise KeyError("bilinmeyen preset %r; mevcut: %s"
                           % (preset, ", ".join(sorted(PRESETS_DIVERSE))))
        cfg = dict(PRESETS_DIVERSE[preset])
        self.preset, self.cfg = preset, cfg

        arch = cfg["arch"]
        if arch == "inception":
            self.backbone = InceptionBackbone(cfg, in_ch, dropout * 0.5)
        elif arch == "hybrid":
            self.backbone = HybridBackbone(cfg, in_ch, dropout)
        else:
            raise KeyError("bilinmeyen arch %r" % arch)

        pooled = self.backbone.out_channels * 2          # ortalama ++ maksimum

        # Ozellik olcekleyici modulun icinde: ONNX kendi kendine yeter, ayri
        # bir scaler dosyasi agirliklardan kopamaz.
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
            nn.Dropout(dropout), nn.Linear(pooled + feat_dim, 128),
            nn.ReLU(inplace=True), nn.Dropout(dropout * 0.5),
            nn.Linear(128, n_classes))

    def set_feature_scaler(self, mean, std):
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
            f = torch.clamp((f - self.feat_mean) / self.feat_std, -10.0, 10.0)
            pooled = torch.cat([pooled, self.feat_branch(f)], dim=1)
        return self.head(pooled)


def build_diverse(preset, dropout=0.2, use_features=True, n_features=37, **kw):
    return DiverseNet(preset, dropout=dropout, use_features=use_features,
                      n_features=n_features, **kw)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="preset'leri listele ve dogrula")
    ap.add_argument("--len", type=int, default=1500)
    ap.add_argument("--features", type=int, default=37)
    args = ap.parse_args()

    print("%-12s %12s  %s" % ("preset", "parametre", "cikti"))
    for name in PRESETS_DIVERSE:
        m = build_diverse(name, n_features=args.features).eval()
        with torch.no_grad():
            out = m(torch.zeros(2, N_LEADS, args.len), torch.zeros(2, args.features))
        n = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print("%-12s %12s  %s" % (name, "{:,}".format(n), tuple(out.shape)))
