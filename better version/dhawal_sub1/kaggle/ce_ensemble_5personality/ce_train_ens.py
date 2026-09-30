"""Train a DIVERSE ENSEMBLE of cross-encoder "personalities" on ``ce_train.parquet`` (er.cross
export-train) - the same input ce_train.py uses. Runs as a Kaggle kernel or as a subprocess of the
single-notebook runner (paths come from env vars, with the Kaggle globs as fallback).

Motivation (Deep-Learning notes, Mitesh Khapra CS7015):
- Dropout "approximates ensembling" and a real ensemble of *decorrelated* models generalises better
  than one bigger model - the antidote to the val->leaderboard gap (a single CE is v6ce; the France
  self-training _fad is what overfit). Each member differs along one DL axis so errors decorrelate:
    * snapshot ensembling via cosine WARM RESTARTS (SGDR)  -> many models from ONE run (S5);
    * a second backbone + fresh head init                  -> architecture/initialisation (S2, S7);
    * dropout level (MC-dropout)                           -> S8;
    * input AUGMENTATION (serialisation swap + char noise) -> data augmentation / denoising AE (S8, S12),
      which also helps zero-shot transfer to France.
- Regularisation (S8): weight decay + label smoothing + grad clip, and a per-country holdout so the
  US-vs-India gap (which predicts France transfer) is visible.

Env: CE_TRAIN (parquet path), CE_MODELS (output dir), CE_PRESET (full|fast|smoke), CE_ONLY (country).
Each snapshot -> <CE_MODELS>/model_<tag>_s<k>/ (weights + tokenizer). Resumable: a member whose
directory already holds config.json is kept, and a personality whose snapshots all exist is skipped.
"""
import glob
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

MAXLEN, BS = 128, 64
XLMR = dict(tag="xlmr", model="xlm-roberta-base", opt="adamw", lr=2e-5, wd=0.01,
            dropout=0.10, aug=0.15, restarts=3, cycle_ep=1, seed=0, ls=0.02)
# 2nd personality: multilingual-e5-base (MIT) = XLM-R architecture with CONTRASTIVE pretraining, i.e. a
# genuinely different initialisation (decorrelated errors) that is fp16-stable. mdeberta-v3 was the
# first choice, but a Kaggle T4 has no bf16 and DeBERTa-v3 is known to diverge (NaN) in fp16.
E5B = dict(tag="e5b", model="intfloat/multilingual-e5-base", opt="adamw", lr=2e-5, wd=0.01,
           dropout=0.20, aug=0.20, restarts=2, cycle_ep=1, seed=1, ls=0.02)
# Presets trade diversity for GPU time (a Kaggle T4 x2 fits "full" in roughly 3-5 h):
#   full  = xlm-roberta (3 snapshots) + e5-base (2 snapshots) -> 5 members
#   fast  = xlm-roberta (2 snapshots)                          -> 2 members
#   smoke = xlm-roberta (1 snapshot, 1500 steps)               -> pipeline check only
PRESETS = {"full": [XLMR, E5B], "fast": [dict(XLMR, restarts=2)],
           "smoke": [dict(XLMR, restarts=1, max_steps=1500)]}
PERSONALITIES = PRESETS[os.environ.get("CE_PRESET", "full")]
ONLY = os.environ.get("CE_ONLY") or None
OUT = os.environ.get("CE_MODELS", "/kaggle/working")
os.makedirs(OUT, exist_ok=True)

path = os.environ.get("CE_TRAIN") or glob.glob("/kaggle/input/**/ce_train.parquet", recursive=True)[0]
full = pd.read_parquet(path)
if ONLY:
    full = full[full.country == ONLY].reset_index(drop=True)
print("loaded", len(full), "pairs | pos rate", round(float(full.label.mean()), 4), flush=True)


def augment(a, b, p, rng):
    """Serialisation swap / field drop / light char noise (S8 data-aug, S12 denoising AE).
    Applied independently to each side of the pair, only with probability p."""
    def one(s):
        if rng.random() >= p:
            return s
        parts = s.split(" | ", 1)
        r = rng.random()
        if len(parts) == 2 and r < 0.34:
            s = parts[1] + " | " + parts[0]                 # address | name
        elif len(parts) == 2 and r < 0.5:
            s = parts[0]                                    # name only (drop address)
        if rng.random() < 0.5 and len(s) > 4:               # delete / swap-adjacent / duplicate a char
            i = rng.randrange(len(s) - 1)
            op = rng.random()
            if op < 0.34:
                s = s[:i] + s[i + 1:]
            elif op < 0.67:
                s = s[:i] + s[i + 1] + s[i] + s[i + 2:]
            else:
                s = s[:i] + s[i] + s[i:]
        return s
    return one(a), one(b)


class Pairs(Dataset):
    def __init__(self, d, aug=0.0, seed=0):
        self.a, self.b = d.a.tolist(), d.b.tolist()
        self.y = d.label.astype(np.float32).tolist()
        self.aug, self.rng = aug, random.Random(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        a, b = self.a[i], self.b[i]
        if self.aug:
            a, b = augment(a, b, self.aug, self.rng)
        return a, b, self.y[i]


def make_collate(tok):
    def collate(batch):
        a, b, y = zip(*batch)
        enc = tok(list(a), list(b), truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(y)
        return enc
    return collate


def build(cfg):
    conf = AutoConfig.from_pretrained(cfg["model"], num_labels=1)
    for k in ("hidden_dropout_prob", "attention_probs_dropout_prob"):
        if hasattr(conf, k):
            setattr(conf, k, cfg["dropout"])
    if hasattr(conf, "classifier_dropout"):
        conf.classifier_dropout = cfg["dropout"]
    return AutoModelForSequenceClassification.from_pretrained(cfg["model"], config=conf)


def evaluate(net, hl, dev):
    net.eval()
    ps, ys, cs = [], [], []
    with torch.no_grad():
        for enc, ct in hl:
            y = enc.pop("labels")
            with torch.autocast("cuda", dtype=torch.float16):
                p = torch.sigmoid(net(**{k: v.to(dev) for k, v in enc.items()}).logits.float().squeeze(-1)).cpu()
            ps.append(p); ys.append(y); cs += ct
    p, y, c = torch.cat(ps).numpy(), torch.cat(ys).numpy(), np.array(cs)

    def ll(m):
        pp, yy = np.clip(p[m], 1e-7, 1 - 1e-7), y[m]
        return float(-np.mean(yy * np.log(pp) + (1 - yy) * np.log(1 - pp))) if m.any() else float("nan")
    net.train()
    return {"logloss": ll(np.ones_like(y, bool)),
            **{f"ll_{k}": ll(c == k) for k in ("US", "India") if (c == k).any()}}


def snap_dir(cfg, k):
    return f"{OUT}/model_{cfg['tag']}_s{k}"


def train_personality(cfg):
    done = [snap_dir(cfg, k) for k in range(cfg["restarts"]) if os.path.exists(snap_dir(cfg, k) + "/config.json")]
    if len(done) == cfg["restarts"]:
        print(f"[{cfg['tag']}] all {len(done)} snapshots exist - skipping", flush=True)
        return {"tag": cfg["tag"], "snapshots": done, "log": [], "seconds": 0, "skipped": True}
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    d = full.sample(frac=1.0, random_state=cfg["seed"]).reset_index(drop=True)
    hold = d.sample(frac=0.02, random_state=cfg["seed"])
    tr = d.drop(hold.index)
    coll = make_collate(tok)
    # per-country holdout so we can watch the India-vs-US gap (predicts France transfer)
    hdl = DataLoader(list(zip(hold.a, hold.b, hold.label.astype(np.float32), hold.country)),
                     batch_size=256, shuffle=False,
                     collate_fn=lambda B: (coll([(a, b, y) for a, b, y, _ in B]), [c for *_, c in B]))
    model = build(cfg).to(dev)
    ng = max(torch.cuda.device_count(), 1)
    net = torch.nn.DataParallel(model) if ng > 1 else model
    dl = DataLoader(Pairs(tr, aug=cfg["aug"], seed=cfg["seed"]), batch_size=BS * ng, shuffle=True,
                    collate_fn=coll, num_workers=2, drop_last=True)
    if cfg["opt"] == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9, nesterov=True, weight_decay=cfg["wd"])
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    max_steps = cfg.get("max_steps") or 0
    steps_ep = min(len(dl), max_steps) if max_steps else len(dl)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=max(cfg["cycle_ep"] * steps_ep, 1))
    scaler = torch.amp.GradScaler("cuda")
    ls, bce = cfg["ls"], torch.nn.BCEWithLogitsLoss()
    t0, log, snaps = time.time(), [], []
    net.train()
    for ep in range(cfg["restarts"] * cfg["cycle_ep"]):
        for i, enc in enumerate(dl):
            if i >= steps_ep:
                break
            y = enc.pop("labels").to(dev)
            yt = y * (1 - ls) + 0.5 * ls                       # label smoothing (S8)
            with torch.autocast("cuda", dtype=torch.float16):
                logit = net(**{k: v.to(dev) for k, v in enc.items()}).logits.float().squeeze(-1)
                loss = bce(logit, yt)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            if i % 500 == 0:
                print(f"[{cfg['tag']}] ep {ep} step {i}/{steps_ep} loss {loss.item():.4f} "
                      f"lr {opt.param_groups[0]['lr']:.2e} {time.time()-t0:.0f}s", flush=True)
        if (ep + 1) % cfg["cycle_ep"] == 0:                    # cosine minimum -> snapshot
            k = len(snaps)
            r = evaluate(net, hdl, dev); r.update(tag=cfg["tag"], snap=k, ep=ep)
            log.append(r)
            # the stacker only sees the MEAN over members, so one diverged member would poison it:
            # keep a snapshot only if its holdout log-loss is finite and clearly better than trivial.
            if not (np.isfinite(r["logloss"]) and r["logloss"] < 0.4):
                print("SKIP bad snapshot (not saved)", r, flush=True)
                continue
            out = snap_dir(cfg, k)
            os.makedirs(out, exist_ok=True)
            model.save_pretrained(out); tok.save_pretrained(out)
            snaps.append(out); print("SNAP", r, flush=True)
    del model, net; torch.cuda.empty_cache()
    return {"tag": cfg["tag"], "snapshots": snaps, "log": log, "seconds": round(time.time() - t0)}


summary = []
for cfg in PERSONALITIES:
    print("\n===== personality", cfg["tag"], cfg["model"], "=====", flush=True)
    summary.append(train_personality(cfg))
json.dump({"personalities": summary, "train_rows": len(full)},
          open(f"{OUT}/train_ens_log.json", "w"), indent=1)
models = [m for p in summary for m in p["snapshots"]]
print("\nDONE. members:", len(models), "->", [os.path.basename(m) for m in models], flush=True)
