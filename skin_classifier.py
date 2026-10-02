"""
Multiclass skin lesion classification on HAM10000 (7 classes) with:
  - lesion-level (leakage-free) train/val/test split
  - class-imbalance handling (weighted loss or weighted sampler)
  - rigorous metrics (balanced acc, macro-F1, per-class AUROC, confusion matrix)
  - Grad-CAM explainability
  - reproducible seeds + JSON result logging

Usage:
  python skin_classifier.py --data_dir ./HAM10000 --backbone efficientnet_b0 --epochs 20
  python skin_classifier.py --data_dir ./HAM10000 --backbone resnet50 --imbalance sampler --seed 1
"""
import argparse, json, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms as T
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (balanced_accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_NAMES = {"akiec": "Actinic keratosis", "bcc": "Basal cell carcinoma",
               "bkl": "Benign keratosis", "df": "Dermatofibroma",
               "mel": "Melanoma", "nv": "Melanocytic nevus", "vasc": "Vascular lesion"}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ----------------------------- data ------------------------------------------
def load_metadata(data_dir):
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "HAM10000_metadata.csv")
    paths = {p.stem: p for p in data_dir.rglob("*.jpg")}
    df["path"] = df["image_id"].map(paths)
    df = df.dropna(subset=["path"]).reset_index(drop=True)
    df["label"] = df["dx"].map({c: i for i, c in enumerate(CLASSES)})
    return df


def lesion_level_split(df, seed, test_size=0.15, val_size=0.15):
    """Split by lesion_id so the same lesion never appears in two splits
    (HAM10000 has multiple images per lesion -> naive splits leak)."""
    g1 = GroupShuffleSplit(1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(g1.split(df, groups=df["lesion_id"]))
    trainval = df.iloc[trainval_idx]
    g2 = GroupShuffleSplit(1, test_size=val_size / (1 - test_size), random_state=seed)
    tr_idx, va_idx = next(g2.split(trainval, groups=trainval["lesion_id"]))
    return trainval.iloc[tr_idx], trainval.iloc[va_idx], df.iloc[test_idx]


class SkinDataset(Dataset):
    def __init__(self, df, transform):
        self.df, self.transform = df.reset_index(drop=True), transform

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        return self.transform(img), int(row["label"])


def get_transforms(size):
    train = T.Compose([
        T.RandomResizedCrop(size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.RandomRotation(30),
        T.ColorJitter(0.2, 0.2, 0.2, 0.02),
        T.ToTensor(), T.Normalize(MEAN, STD)])
    evalt = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])
    return train, evalt


# ----------------------------- model -----------------------------------------
def build_model(name, n_classes=7):
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
        target = m.layer4[-1]
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
        target = m.features[-1]
    else:
        raise ValueError(name)
    return m, target


# ----------------------------- Grad-CAM --------------------------------------
class GradCAM:
    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        layer.register_forward_hook(self._fwd)

    def _fwd(self, _, __, out):
        self.acts = out
        out.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x, cls=None):
        self.model.eval(); self.model.zero_grad()
        logits = self.model(x)
        cls = logits.argmax(1) if cls is None else cls
        logits[0, cls].sum().backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam[0, 0].detach().cpu().numpy(), int(cls)


def save_gradcam(model, layer, dataset, device, out_path, n=8):
    cam = GradCAM(model, layer)
    idx = np.random.choice(len(dataset), n, replace=False)
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.8))
    for j, i in enumerate(idx):
        x, y = dataset[i]
        heat, pred = cam(x.unsqueeze(0).to(device).requires_grad_(True))
        img = (x.permute(1, 2, 0).numpy() * STD + MEAN).clip(0, 1)
        axes[0, j].imshow(img); axes[0, j].axis("off")
        axes[0, j].set_title(f"true: {CLASSES[y]}", fontsize=8)
        axes[1, j].imshow(img); axes[1, j].imshow(heat, cmap="jet", alpha=0.45)
        axes[1, j].axis("off"); axes[1, j].set_title(f"pred: {CLASSES[pred]}", fontsize=8)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


# ----------------------------- train / eval ----------------------------------
@torch.no_grad()
def predict(model, loader, device):
    model.eval(); probs, ys = [], []
    for x, y in loader:
        probs.append(F.softmax(model(x.to(device)), 1).cpu()); ys.append(y)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


def metrics(probs, ys):
    preds = probs.argmax(1)
    aucs = {}
    for k, c in enumerate(CLASSES):
        if (ys == k).any():
            aucs[c] = float(roc_auc_score((ys == k).astype(int), probs[:, k]))
    return {"accuracy": float((preds == ys).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(ys, preds)),
            "macro_f1": float(f1_score(ys, preds, average="macro")),
            "macro_auroc": float(np.mean(list(aucs.values()))),
            "per_class_auroc": aucs}


def plot_confusion(ys, preds, path):
    cm = confusion_matrix(ys, preds, labels=range(len(CLASSES)), normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(7)); ax.set_xticklabels(CLASSES, rotation=45)
    ax.set_yticks(range(7)); ax.set_yticklabels(CLASSES)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Normalized confusion matrix")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--backbone", default="efficientnet_b0", choices=["resnet50", "efficientnet_b0"])
    ap.add_argument("--imbalance", default="loss", choices=["none", "loss", "sampler"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="runs")
    a = ap.parse_args()

    set_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out_dir) / f"{a.backbone}_{a.imbalance}_s{a.seed}"
    out.mkdir(parents=True, exist_ok=True)

    df = load_metadata(a.data_dir)
    tr, va, te = lesion_level_split(df, a.seed)
    print(f"images: train {len(tr)} | val {len(va)} | test {len(te)}")
    tr_t, ev_t = get_transforms(a.img_size)
    ds_tr, ds_va, ds_te = SkinDataset(tr, tr_t), SkinDataset(va, ev_t), SkinDataset(te, ev_t)

    counts = np.bincount(tr["label"], minlength=7)
    cls_w = torch.tensor(len(tr) / (7 * counts), dtype=torch.float32)
    sampler = None
    if a.imbalance == "sampler":
        sampler = WeightedRandomSampler((1.0 / counts)[tr["label"].values], len(tr), replacement=True)
    dl_tr = DataLoader(ds_tr, a.batch_size, shuffle=sampler is None, sampler=sampler,
                       num_workers=4, pin_memory=True)
    dl_va = DataLoader(ds_va, 64, num_workers=4)
    dl_te = DataLoader(ds_te, 64, num_workers=4)

    model, target_layer = build_model(a.backbone)
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=cls_w.to(device) if a.imbalance == "loss" else None)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    best_f1, history = -1, []
    for ep in range(a.epochs):
        model.train(); total = 0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            total += loss.item() * len(y)
        sched.step()
        m = metrics(*predict(model, dl_va, device))
        history.append({"epoch": ep + 1, "train_loss": total / len(tr), **{k: v for k, v in m.items() if k != "per_class_auroc"}})
        print(f"ep {ep+1:02d} loss {total/len(tr):.4f} | val bal-acc {m['balanced_accuracy']:.3f} macro-F1 {m['macro_f1']:.3f}")
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]; torch.save(model.state_dict(), out / "best.pt")

    # final test evaluation with best (val-selected) checkpoint
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    probs, ys = predict(model, dl_te, device)
    res = metrics(probs, ys)
    res["args"] = vars(a); res["history"] = history
    res["report"] = classification_report(ys, probs.argmax(1), target_names=CLASSES, output_dict=True, zero_division=0)
    json.dump(res, open(out / "results.json", "w"), indent=2)
    plot_confusion(ys, probs.argmax(1), out / "confusion_matrix.png")
    save_gradcam(model, target_layer, ds_te, device, out / "gradcam_samples.png")
    print("\nTEST:", {k: round(v, 4) for k, v in res.items() if isinstance(v, float)})
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
