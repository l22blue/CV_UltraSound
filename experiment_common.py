from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
try:
    import torch_directml
    HAS_DML = True
    print("[INFO] torch-directml found. AMD/Intel GPU acceleration is available.")
except ImportError:
    HAS_DML = False
    print("[INFO] torch-directml not found. Falling back to default devices.")
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as T


ROOT_DEFAULT = r"c:\Univ\학부 연구생\컴퓨터 비전 스터디\prof_seo_jg_USG_hypoE (아산병원 서준교 교수님 초음파)"


@dataclass(frozen=True)
class SplitSpec:
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15

    def validate(self) -> None:
        s = self.train + self.val + self.test
        if not math.isclose(s, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"Split must sum to 1.0, got {s}")


@dataclass(frozen=True)
class ExperimentConfig:
    root: str
    excel_path: str
    images_dir: str
    label_col: str = "Diagnosis(Cancer)"
    id_col: str = "ID"
    split: SplitSpec = SplitSpec()
    seed: int = 42
    epochs: int = 1
    batch_size: int = 16
    lr: float = 1e-3
    num_workers: int = 0
    preprocessing: str = "none"
    device: str = "cpu"
    output_dir: str = "outputs"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def list_patient_images(images_dir: Path) -> Dict[int, List[Path]]:
    pat = re.compile(r"^pt_(\d+)_(\d+)\.jpg$", re.IGNORECASE)
    out: Dict[int, List[Path]] = {}
    for name in os.listdir(images_dir):
        m = pat.match(name)
        if not m:
            continue
        pid = int(m.group(1))
        out.setdefault(pid, []).append(images_dir / name)
    for pid in out:
        out[pid].sort()
    return out


def load_labels(excel_path: Path, id_col: str, label_col: str) -> Dict[int, int]:
    df = pd.read_excel(excel_path)
    labels: Dict[int, int] = {}
    for _, row in df[[id_col, label_col]].iterrows():
        labels[int(row[id_col])] = int(row[label_col])
    return labels


def split_patients(pids: Sequence[int], split: SplitSpec, seed: int) -> Tuple[List[int], List[int], List[int]]:
    split.validate()
    pids = list(pids)
    random.Random(seed).shuffle(pids)
    n = len(pids)
    n_train = int(round(n * split.train))
    n_val = int(round(n * split.val))
    n_train = min(max(n_train, 1), n - 2)
    n_val = min(max(n_val, 1), n - n_train - 1)
    return pids[:n_train], pids[n_train : n_train + n_val], pids[n_train + n_val :]


def make_preprocess(name: str) -> Callable[[np.ndarray], np.ndarray]:
    name = name.lower().strip()
    if name == "none":
        return lambda x: x
    if name == "gaussian":
        return lambda x: cv2.GaussianBlur(x, (5, 5), 0)
    if name == "median":
        return lambda x: cv2.medianBlur(x, 5)
    if name == "bilateral":
        return lambda x: cv2.bilateralFilter(x, d=9, sigmaColor=75, sigmaSpace=75)
    if name in {"non_local_means", "nlm"}:
        return lambda x: cv2.fastNlMeansDenoising(x, None, h=10, templateWindowSize=7, searchWindowSize=21)
    raise ValueError("Unknown preprocessing")


class ImageDataset(Dataset):
    def __init__(self, items, preprocess, transform) -> None:
        self.items = list(items)
        self.preprocess = preprocess
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx):
        path, label, pid = self.items[idx]
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        img = self.preprocess(img)
        img = torch.from_numpy(img).unsqueeze(0).to(torch.uint8)
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32), pid


def build_transforms():
    return T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Resize((224, 224), interpolation=InterpolationMode.BILINEAR, antialias=True),
            T.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


def build_model():
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m


def make_items(pids, patient_to_images, labels):
    items = []
    for pid in pids:
        for p in patient_to_images[pid]:
            items.append((p, labels[pid], pid))
    return items


@torch.no_grad()
def predict_patient_probs(model, loader, device):
    model.eval()
    probs: Dict[int, List[float]] = {}
    labels: Dict[int, int] = {}
    for x, y, pid in loader:
        logits = model(x.to(device)).squeeze(1)
        p = torch.sigmoid(logits).detach().cpu().numpy().tolist()
        y = y.numpy().astype(int).tolist()
        pid = pid.numpy().astype(int).tolist()
        for pi, yi, p_i in zip(pid, y, p, strict=True):
            probs.setdefault(int(pi), []).append(float(p_i))
            labels[int(pi)] = int(yi)
    return {k: float(np.mean(v)) for k, v in probs.items()}, labels


def patient_metrics(patient_probs, patient_labels):
    pids = sorted(patient_labels.keys())
    y_true = np.array([patient_labels[k] for k in pids], dtype=np.int32)
    y_prob = np.array([patient_probs[k] for k in pids], dtype=np.float32)
    y_pred = (y_prob >= 0.5).astype(np.int32)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {
        "patients": float(len(pids)),
        "accuracy": float(np.mean(y_true == y_pred)),
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan"),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def run_experiment(preprocessing: str, root: str, epochs: int, batch_size: int, seed: int, lr: float, num_workers: int) -> None:
    root_path = Path(root)
    cfg = ExperimentConfig(
        root=str(root_path),
        excel_path=str(root_path / "TRUS_AI_CRF_forshare.xlsx"),
        images_dir=str(root_path / "processed"),
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        num_workers=num_workers,
        preprocessing=preprocessing,
        device="dml" if HAS_DML else ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    set_seed(cfg.seed)

    labels = load_labels(Path(cfg.excel_path), cfg.id_col, cfg.label_col)
    patient_to_images = list_patient_images(Path(cfg.images_dir))
    pids = sorted(set(labels.keys()) & set(patient_to_images.keys()))
    train_p, val_p, test_p = split_patients(pids, cfg.split, cfg.seed)

    preprocess = make_preprocess(cfg.preprocessing)
    transform = build_transforms()
    train_ds = ImageDataset(make_items(train_p, patient_to_images, labels), preprocess, transform)
    val_ds = ImageDataset(make_items(val_p, patient_to_images, labels), preprocess, transform)
    test_ds = ImageDataset(make_items(test_p, patient_to_images, labels), preprocess, transform)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    if cfg.device == "dml":
        device = torch_directml.device()
        print(f"[INFO] Using AMD/Intel GPU via DirectML: {torch_directml.device_name(device.index)}")
    else:
        device = torch.device(cfg.device)
        print(f"[INFO] Using device: {device}")

    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_acc = -1.0
    best_state = None
    start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for x, y, _pid in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x).squeeze(1), y)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))

        val_probs, val_labels = predict_patient_probs(model, val_loader, device)
        val_acc = patient_metrics(val_probs, val_labels)["accuracy"]
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:02d}/{cfg.epochs} train_loss={np.mean(losses):.4f} val_acc={val_acc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_probs, test_labels = predict_patient_probs(model, test_loader, device)
    m = patient_metrics(test_probs, test_labels)
    elapsed = time.time() - start

    out_dir = root_path / cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"result_{cfg.preprocessing}_seed{cfg.seed}_{stamp}.json"
    out_path.write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "split_counts": {"train_patients": len(train_p), "val_patients": len(val_p), "test_patients": len(test_p)},
                "test_metrics_patient_level": m,
                "elapsed_sec": elapsed,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("saved", out_path)
    print("test metrics:", json.dumps(m, ensure_ascii=False))


def parse_common_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=ROOT_DEFAULT)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()
