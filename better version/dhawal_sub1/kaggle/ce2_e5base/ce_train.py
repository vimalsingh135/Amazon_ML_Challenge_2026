"""Kaggle GPU kernel: fine-tune a multilingual cross-encoder (xlm-roberta-base, MIT, 278M) on raw
"name | address" pairs (FIT fold-0 S1 only, exported by er.cross export-train).
Output: /kaggle/working/model (weights + tokenizer) and train_log.json."""
import glob
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

MODEL, MAXLEN, BS, LR, EPOCHS, SEED = "intfloat/multilingual-e5-base", 128, 64, 2e-5, 2, 0   # second opinion
ONLY = None   # e.g. "US": train on one country only (transfer test); the loco kernel sets it
torch.manual_seed(SEED)
np.random.seed(SEED)
path = glob.glob("/kaggle/input/**/ce_train.parquet", recursive=True)[0]
df = pd.read_parquet(path)
if ONLY:
    df = df[df.country == ONLY].reset_index(drop=True)
df = df[["a", "b", "label"]]
hold = df.sample(frac=0.02, random_state=SEED)
df = df.drop(hold.index)
print("train", len(df), "hold", len(hold), "pos rate", df.label.mean(), flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)


class Pairs(Dataset):
    def __init__(self, d):
        self.a, self.b, self.y = d.a.tolist(), d.b.tolist(), d.label.astype(np.float32).tolist()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.a[i], self.b[i], self.y[i]


def collate(batch):
    a, b, y = zip(*batch)
    enc = tok(list(a), list(b), truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
    enc["labels"] = torch.tensor(y)
    return enc


dev = "cuda"
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).to(dev)
ngpu = torch.cuda.device_count()
net = torch.nn.DataParallel(model) if ngpu > 1 else model
dl = DataLoader(Pairs(df), batch_size=BS * max(ngpu, 1), shuffle=True, collate_fn=collate, num_workers=2)
hl = DataLoader(Pairs(hold), batch_size=256, shuffle=False, collate_fn=collate, num_workers=2)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps = EPOCHS * len(dl)
sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
scaler = torch.cuda.amp.GradScaler()
bce = torch.nn.BCEWithLogitsLoss()


def evaluate():
    net.eval()
    ps, ys = [], []
    with torch.no_grad():
        for enc in hl:
            y = enc.pop("labels")
            with torch.cuda.amp.autocast():
                ps.append(torch.sigmoid(net(**{k: v.to(dev) for k, v in enc.items()}).logits.float().squeeze(-1)).cpu())
            ys.append(y)
    p, y = torch.cat(ps).numpy(), torch.cat(ys).numpy()
    ll = float(-np.mean(y * np.log(np.clip(p, 1e-7, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-7, 1))))
    acc = float(((p > 0.5) == (y > 0.5)).mean())
    net.train()
    return {"logloss": ll, "acc": acc}


t0, log = time.time(), []
net.train()
for ep in range(EPOCHS):
    for i, enc in enumerate(dl):
        y = enc.pop("labels").to(dev)
        with torch.cuda.amp.autocast():
            logit = net(**{k: v.to(dev) for k, v in enc.items()}).logits.float().squeeze(-1)
            loss = bce(logit, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sch.step()
        if i % 500 == 0:
            print(f"ep {ep} step {i}/{len(dl)} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        if i and i % 5000 == 0:
            r = evaluate(); r["step"] = i; log.append(r); print("hold", r, flush=True)
r = evaluate(); r["step"] = "final"; log.append(r); print("hold final", r, flush=True)
os.makedirs("/kaggle/working/model", exist_ok=True)
model.save_pretrained("/kaggle/working/model")
tok.save_pretrained("/kaggle/working/model")
json.dump({"log": log, "train_rows": len(df), "seconds": time.time() - t0, "model": MODEL}, open("/kaggle/working/train_log.json", "w"), indent=1)
print("saved", flush=True)
