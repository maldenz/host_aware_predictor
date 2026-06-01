from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from host_aware_predictor.models.host_unaware import HostUnawareConfig, HostUnawarePredictor


class EmbeddingRegressionDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    def __init__(self, embeddings: torch.Tensor, targets: torch.Tensor) -> None:
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got {tuple(embeddings.shape)}")
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)
        if targets.ndim != 2:
            raise ValueError(f"targets must be 1D or 2D, got {tuple(targets.shape)}")
        if embeddings.shape[0] != targets.shape[0]:
            raise ValueError(
                f"embedding/target row mismatch: {embeddings.shape[0]} vs {targets.shape[0]}"
            )
        self.embeddings = embeddings.float()
        self.targets = targets.float()

    def __len__(self) -> int:
        return int(self.embeddings.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        return self.embeddings[index], self.targets[index], index


def _load_torch_payload(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_embedding_payload(embedding_dir: Path, split: str) -> dict[str, Any]:
    path = embedding_dir / f"{split}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cached embedding file: {path}. Run scripts/cache_nt_embeddings_dragonn.py first."
        )
    payload = _load_torch_payload(path)
    if "embeddings" not in payload or "targets" not in payload:
        raise KeyError(f"{path} must contain 'embeddings' and 'targets'.")
    return payload


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _auto_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_hidden_dims(value: str) -> tuple[int, ...]:
    if value.strip() == "":
        return tuple()
    return tuple(int(dim.strip()) for dim in value.split(",") if dim.strip())


def standardize_from_train(
    train_embeddings: torch.Tensor,
    *other_embeddings: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, torch.Tensor]:
    mean = train_embeddings.mean(dim=0, keepdim=True)
    std = train_embeddings.std(dim=0, keepdim=True).clamp_min(1e-6)
    standardized_train = (train_embeddings - mean) / std
    standardized_other = [(tensor - mean) / std for tensor in other_embeddings]
    return standardized_train, standardized_other, mean, std


def _safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    if y_true.size < 2:
        return float("nan")
    true_std = y_true.std()
    pred_std = y_pred.std()
    if true_std == 0 or pred_std == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    residual = y_pred - y_true
    mse_per_target = np.mean(residual**2, axis=0)
    mae_per_target = np.mean(np.abs(residual), axis=0)

    y_mean = np.mean(y_true, axis=0, keepdims=True)
    ss_res = np.sum(residual**2, axis=0)
    ss_tot = np.sum((y_true - y_mean) ** 2, axis=0)
    r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)

    pearson = np.array([
        _safe_pearson(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])
    ])

    return {
        "mse": float(np.mean(mse_per_target)),
        "mae": float(np.mean(mae_per_target)),
        "r2_macro": float(np.nanmean(r2)),
        "pearson_macro": float(np.nanmean(pearson)),
    }


def evaluate(
    model: nn.Module,
    data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor, int]],
    device: torch.device,
    loss_fn: nn.Module,
) -> tuple[float, dict[str, float], np.ndarray, np.ndarray, list[int]]:
    model.eval()
    losses: list[float] = []
    y_true_batches: list[np.ndarray] = []
    y_pred_batches: list[np.ndarray] = []
    indices: list[int] = []

    with torch.inference_mode():
        for x, y, batch_indices in data_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            losses.append(float(loss.item()) * x.shape[0])
            y_true_batches.append(y.detach().cpu().numpy())
            y_pred_batches.append(pred.detach().cpu().numpy())
            indices.extend(int(i) for i in batch_indices)

    y_true = np.concatenate(y_true_batches, axis=0)
    y_pred = np.concatenate(y_pred_batches, axis=0)
    metrics = compute_regression_metrics(y_true, y_pred)
    loss_mean = float(sum(losses) / max(1, len(data_loader.dataset)))
    return loss_mean, metrics, y_true, y_pred, indices


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_predictions(
    path: Path,
    sequence_ids: list[str],
    target_names: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sequence_id"]
    for name in target_names:
        fieldnames.append(f"true_{name}")
    for name in target_names:
        fieldnames.append(f"pred_{name}")

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_idx, sequence_id in enumerate(sequence_ids):
            row: dict[str, Any] = {"sequence_id": sequence_id}
            for target_idx, name in enumerate(target_names):
                row[f"true_{name}"] = float(y_true[row_idx, target_idx])
            for target_idx, name in enumerate(target_names):
                row[f"pred_{name}"] = float(y_pred[row_idx, target_idx])
            writer.writerow(row)


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    config: HostUnawareConfig,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    metrics: dict[str, Any],
    target_names: list[str],
    embedding_dir: Path,
    standardize: bool,
    embedding_mean: torch.Tensor | None,
    embedding_std: torch.Tensor | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(config),
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "target_names": target_names,
            "embedding_dir": str(embedding_dir),
            "standardize": standardize,
            "embedding_mean": embedding_mean.cpu() if embedding_mean is not None else None,
            "embedding_std": embedding_std.cpu() if embedding_std is not None else None,
        },
        path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train host-unaware MPRA baseline head on cached NT sequence embeddings."
    )
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dims", default="512,128")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", default="gelu")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = _auto_device(args.device)

    train_payload = load_embedding_payload(args.embedding_dir, "train")
    valid_payload = load_embedding_payload(args.embedding_dir, "valid")
    test_payload = load_embedding_payload(args.embedding_dir, "test")

    train_x = train_payload["embeddings"].float()
    valid_x = valid_payload["embeddings"].float()
    test_x = test_payload["embeddings"].float()
    train_y = train_payload["targets"].float()
    valid_y = valid_payload["targets"].float()
    test_y = test_payload["targets"].float()

    standardize = not args.no_standardize
    embedding_mean = None
    embedding_std = None
    if standardize:
        train_x, others, embedding_mean, embedding_std = standardize_from_train(train_x, valid_x, test_x)
        valid_x, test_x = others

    target_names = list(train_payload.get("target_names") or [f"target_{i}" for i in range(train_y.shape[1])])
    if len(target_names) != train_y.shape[1]:
        target_names = [f"target_{i}" for i in range(train_y.shape[1])]

    train_dataset = EmbeddingRegressionDataset(train_x, train_y)
    valid_dataset = EmbeddingRegressionDataset(valid_x, valid_y)
    test_dataset = EmbeddingRegressionDataset(test_x, test_y)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    config = HostUnawareConfig(
        input_dim=int(train_x.shape[1]),
        output_dim=int(train_y.shape[1]),
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        dropout=float(args.dropout),
        activation=args.activation,
    )
    model = HostUnawarePredictor(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_rel = Path("host_unaware") / "mpra_dragonn" / "nt_sequence_baseline" / run_name
    report_dir = args.reports_dir / run_rel
    checkpoint_dir = args.checkpoints_dir / run_rel
    report_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config_payload: dict[str, Any] = {
        "run_name": run_name,
        "baseline": "host_unaware_nt_sequence_only",
        "embedding_dir": str(args.embedding_dir),
        "reports_dir": str(report_dir),
        "checkpoints_dir": str(checkpoint_dir),
        "model_config": asdict(config),
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "seed": args.seed,
            "device": str(device),
            "standardize": standardize,
        },
        "data": {
            "train_shape": list(train_x.shape),
            "valid_shape": list(valid_x.shape),
            "test_shape": list(test_x.shape),
            "target_shape": list(train_y.shape),
            "target_names": target_names,
        },
    }
    (report_dir / "config.json").write_text(json.dumps(config_payload, indent=2, sort_keys=True))

    print(f"Run: {run_name}")
    print(f"Reports: {report_dir}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Device: {device}")
    print(f"Train embeddings: {tuple(train_x.shape)} targets: {tuple(train_y.shape)}")

    best_valid_mse = math.inf
    best_epoch = -1
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_total = 0.0
        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_loss_total += float(loss.item()) * x.shape[0]

        train_loss = train_loss_total / max(1, len(train_dataset))
        valid_loss, valid_metrics, _, _, _ = evaluate(model, valid_loader, device, loss_fn)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_mse": valid_metrics["mse"],
            "valid_mae": valid_metrics["mae"],
            "valid_r2_macro": valid_metrics["r2_macro"],
            "valid_pearson_macro": valid_metrics["pearson_macro"],
        }
        history.append(row)
        write_history(report_dir / "history.csv", history)

        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6g} "
            f"valid_mse={valid_metrics['mse']:.6g} "
            f"valid_pearson={valid_metrics['pearson_macro']:.4f}"
        )

        save_checkpoint(
            checkpoint_dir / "last.pt",
            model=model,
            config=config,
            epoch=epoch,
            optimizer=optimizer,
            metrics={"valid": valid_metrics, "train_loss": train_loss},
            target_names=target_names,
            embedding_dir=args.embedding_dir,
            standardize=standardize,
            embedding_mean=embedding_mean,
            embedding_std=embedding_std,
        )

        if valid_metrics["mse"] < best_valid_mse:
            best_valid_mse = valid_metrics["mse"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model=model,
                config=config,
                epoch=epoch,
                optimizer=optimizer,
                metrics={"valid": valid_metrics, "train_loss": train_loss},
                target_names=target_names,
                embedding_dir=args.embedding_dir,
                standardize=standardize,
                embedding_mean=embedding_mean,
                embedding_std=embedding_std,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                break

    best_payload = _load_torch_payload(checkpoint_dir / "best.pt")
    model.load_state_dict(best_payload["model_state_dict"])

    valid_loss, valid_metrics, valid_true, valid_pred, valid_indices = evaluate(
        model,
        valid_loader,
        device,
        loss_fn,
    )
    test_loss, test_metrics, test_true, test_pred, test_indices = evaluate(
        model,
        test_loader,
        device,
        loss_fn,
    )

    valid_sequence_ids_all = list(valid_payload.get("sequence_ids") or [f"valid_{i}" for i in range(len(valid_dataset))])
    test_sequence_ids_all = list(test_payload.get("sequence_ids") or [f"test_{i}" for i in range(len(test_dataset))])
    valid_sequence_ids = [valid_sequence_ids_all[i] for i in valid_indices]
    test_sequence_ids = [test_sequence_ids_all[i] for i in test_indices]

    write_predictions(
        report_dir / "predictions_valid.csv",
        valid_sequence_ids,
        target_names,
        valid_true,
        valid_pred,
    )
    write_predictions(
        report_dir / "predictions_test.csv",
        test_sequence_ids,
        target_names,
        test_true,
        test_pred,
    )

    metrics_payload = {
        "best_epoch": best_epoch,
        "valid": {"loss": valid_loss, **valid_metrics},
        "test": {"loss": test_loss, **test_metrics},
        "checkpoint_best": str(checkpoint_dir / "best.pt"),
        "checkpoint_last": str(checkpoint_dir / "last.pt"),
        "reports": {
            "history": str(report_dir / "history.csv"),
            "valid_predictions": str(report_dir / "predictions_valid.csv"),
            "test_predictions": str(report_dir / "predictions_test.csv"),
        },
    }
    (report_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, sort_keys=True))

    print("Final metrics:")
    print(json.dumps(metrics_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
