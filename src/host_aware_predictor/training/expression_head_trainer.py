"""Head-agnostic training utilities for host-aware expression prediction."""

from __future__ import annotations

import json
import math
import random
import re
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from host_aware_predictor.models.fusion_heads import build_expression_head, expression_head_config_dict

from .element_quantification_dataset import (
    ElementQuantificationDataset,
    collate_element_batch,
    discover_conditions,
    discover_quantification_files,
    load_quantification_table,
    make_records_for_split,
    read_split_names,
)
from .metrics import regression_metrics




def safe_run_name(value: str) -> str:
    """Return a filesystem-safe run/report name component."""

    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value)
    value = value.strip("._-")
    if not value:
        raise ValueError("Run name cannot be empty after sanitization.")
    return value


def ensure_run_name(args: Any) -> None:
    """Set args.run_name to the default timestamp+head prefix when absent."""

    run_name = getattr(args, "run_name", None)
    if run_name is None:
        timestamp_format = getattr(args, "timestamp_format", "%Y%m%d-%H%M%S")
        run_name = f"{datetime.now().strftime(timestamp_format)}_{args.head}"
    args.run_name = safe_run_name(run_name)


def run_output_path(args: Any, suffix: str) -> Path:
    """Path for a run-scoped artifact named <run_name>_<suffix>."""

    return Path(args.output_dir) / f"{args.run_name}_{suffix}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_serializable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    if hasattr(value, "__dict__") and value.__class__.__module__ != "builtins":
        return to_serializable(value.__dict__)
    return value


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(obj), handle, indent=2, allow_nan=True)


def resolve_paths(args: Any) -> None:
    """Fill derived path arguments in-place."""

    ensure_run_name(args)

    processed = Path(args.processed_dir)
    args.processed_dir = processed
    args.dna_embedding_dir = Path(args.dna_embedding_dir) if args.dna_embedding_dir else processed / "dna_emb" / "by_name"
    args.host_embedding_dir = Path(args.host_embedding_dir) if args.host_embedding_dir else processed / "host_emb"
    args.split_dir = Path(args.split_dir) if args.split_dir else processed / "splits"

    if getattr(args, "condition", None) is not None and getattr(args, "conditions", None) is not None:
        raise ValueError("Pass either --condition or --conditions, not both.")
    if getattr(args, "condition", None) is not None:
        args.conditions = [args.condition]
    elif getattr(args, "conditions", None) == []:
        args.conditions = None

    if getattr(args, "quantification_tsv", None):
        args.quantification_tsv = [Path(path) for path in args.quantification_tsv]
    else:
        discovery_condition = args.conditions[0] if args.conditions and len(args.conditions) == 1 else None
        args.quantification_tsv = discover_quantification_files(processed, condition=discovery_condition)

    if getattr(args, "output_dir", None) is None:
        if args.conditions:
            condition_label = "_".join(str(condition) for condition in args.conditions)
        else:
            condition_label = "all_conditions"
        args.output_dir = Path("runs") / f"{args.head}_head" / f"{condition_label}_{args.target_col}" / args.run_name
    else:
        args.output_dir = Path(args.output_dir)

def enforce_host_specific_baseline(args: Any) -> None:
    """Require sequence-only baseline training to use exactly one condition."""

    if getattr(args, "head", None) != "sequence_only":
        return

    conditions = getattr(args, "conditions", None)
    if conditions is None or len(conditions) != 1:
        raise ValueError(
            "The sequence_only baseline requires exactly one cell: "
            "pass --condition <cell> or --conditions <cell>."
        )

def make_datasets(args: Any) -> tuple[dict[str, ElementQuantificationDataset], dict[str, Any], pd.DataFrame]:
    known_conditions = discover_conditions(args.host_embedding_dir)
    df = load_quantification_table(
        args.quantification_tsv,
        condition=None,
        known_conditions=known_conditions,
        name_col=args.name_col,
        condition_col=args.condition_col,
        target_col=args.target_col,
        weight_col=args.weight_col,
    )

    if args.conditions:
        requested_conditions = {str(condition) for condition in args.conditions}
        df = df.loc[df[args.condition_col].astype(str).isin(requested_conditions)].copy()
        if df.empty:
            available = sorted(set(load_quantification_table(
                args.quantification_tsv,
                condition=None,
                known_conditions=known_conditions,
                name_col=args.name_col,
                condition_col=args.condition_col,
                target_col=args.target_col,
                weight_col=args.weight_col,
            )[args.condition_col].astype(str)))
            raise RuntimeError(
                f"No rows left after filtering to --conditions={sorted(requested_conditions)}. "
                f"Available conditions in quantification tables: {available}"
            )

    split_names = {
        "train": read_split_names(args.split_dir / "train_names.txt"),
        "val": read_split_names(args.split_dir / "val_names.txt"),
        "test": read_split_names(args.split_dir / "test_names.txt"),
    }

    records: dict[str, list[Any]] = {}
    reports: dict[str, Any] = {}
    for split, names in split_names.items():
        split_records, report = make_records_for_split(
            df,
            split=split,
            split_names=names,
            dna_embedding_dir=args.dna_embedding_dir,
            host_embedding_dir=args.host_embedding_dir,
            name_col=args.name_col,
            condition_col=args.condition_col,
            target_col=args.target_col,
            weight_col=args.weight_col,
            require_dna_embedding=True,
            require_host_embedding=True,
        )
        records[split] = split_records
        reports[split] = report

    datasets = {
        split: ElementQuantificationDataset(
            split_records,
            dna_pooling=args.dna_pooling,
            host_pooling=args.host_pooling,
            cache_embeddings=not args.no_cache_embeddings,
        )
        for split, split_records in records.items()
    }

    if len(datasets["train"]) == 0:
        raise RuntimeError("The train split has zero usable records. Check split names, quantification tables, and embedding paths.")
    if len(datasets["val"]) == 0:
        raise RuntimeError("The val split has zero usable records. Check split names, quantification tables, and embedding paths.")

    return datasets, reports, df


def make_loader(
    dataset: ElementQuantificationDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        collate_fn=collate_element_batch,
    )


def original_scale(values: torch.Tensor, *, mean: float, std: float) -> torch.Tensor:
    return values * std + mean


def transformed_target(values: torch.Tensor, *, mean: float, std: float) -> torch.Tensor:
    return (values - mean) / std


def extract_expression(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    expression = getattr(output, "expression", None)
    if torch.is_tensor(expression):
        return expression
    raise TypeError(f"Model output must be a Tensor or have Tensor .expression, got {type(output).__name__}.")


def loss_from_batch(
    model: nn.Module,
    batch: dict[str, object],
    *,
    criterion: nn.Module,
    device: torch.device,
    use_weights: bool,
    target_mean: float,
    target_std: float,
    autocast_enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequence_embedding = batch["sequence_embedding"].to(device=device, dtype=torch.float32, non_blocking=True)
    host_embedding = batch["host_embedding"].to(device=device, dtype=torch.float32, non_blocking=True)
    target = batch["target"].to(device=device, dtype=torch.float32, non_blocking=True)
    target_for_loss = transformed_target(target, mean=target_mean, std=target_std)

    ctx = torch.autocast(device_type=device.type, enabled=autocast_enabled) if device.type in {"cuda", "cpu"} else nullcontext()
    with ctx:
        pred_for_loss = extract_expression(model(sequence_embedding, host_embedding))
        loss_vec = criterion(pred_for_loss, target_for_loss)
        if use_weights:
            weight = batch["weight"].to(device=device, dtype=loss_vec.dtype, non_blocking=True)
            denom = torch.clamp(weight.sum(), min=1.0)
            loss = (loss_vec * weight).sum() / denom
        else:
            loss = loss_vec.mean()

    pred_original = original_scale(pred_for_loss.detach().float(), mean=target_mean, std=target_std)
    return loss, pred_original, target.detach().float()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    criterion: nn.Module,
    device: torch.device,
    use_weights: bool,
    target_mean: float,
    target_std: float,
    autocast_enabled: bool,
    grad_clip_norm: float,
) -> dict[str, float]:
    model.train()
    losses: list[float] = []
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        loss, pred, target = loss_from_batch(
            model,
            batch,
            criterion=criterion,
            device=device,
            use_weights=use_weights,
            target_mean=target_mean,
            target_std=target_std,
            autocast_enabled=autocast_enabled,
        )

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        losses.append(float(loss.detach().cpu()))
        all_true.append(target.cpu().numpy().reshape(-1))
        all_pred.append(pred.cpu().numpy().reshape(-1))

    metrics = regression_metrics(np.concatenate(all_true), np.concatenate(all_pred))
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    criterion: nn.Module,
    device: torch.device,
    use_weights: bool,
    target_mean: float,
    target_std: float,
    autocast_enabled: bool,
    return_predictions: bool = False,
) -> tuple[dict[str, float], pd.DataFrame | None]:
    model.eval()
    losses: list[float] = []
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_names: list[str] = []
    all_conditions: list[str] = []

    for batch in loader:
        loss, pred, target = loss_from_batch(
            model,
            batch,
            criterion=criterion,
            device=device,
            use_weights=use_weights,
            target_mean=target_mean,
            target_std=target_std,
            autocast_enabled=autocast_enabled,
        )
        losses.append(float(loss.detach().cpu()))
        true_np = target.cpu().numpy().reshape(-1)
        pred_np = pred.cpu().numpy().reshape(-1)
        all_true.append(true_np)
        all_pred.append(pred_np)
        if return_predictions:
            all_names.extend(batch["name"])
            all_conditions.extend(batch["condition"])

    y_true = np.concatenate(all_true) if all_true else np.array([], dtype=np.float64)
    y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=np.float64)
    metrics = regression_metrics(y_true, y_pred)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")

    predictions = None
    if return_predictions:
        predictions = pd.DataFrame(
            {
                "name": all_names,
                "condition": all_conditions,
                "y_true": y_true,
                "y_pred": y_pred,
                "residual": y_pred - y_true,
            }
        )
    return metrics, predictions


def build_model_from_args(args: Any, *, sequence_embedding_dim: int, host_embedding_dim: int) -> nn.Module:
    try:
        extra_head_kwargs = json.loads(getattr(args, "head_kwargs_json", "{}") or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"--head-kwargs-json must be valid JSON: {exc}") from exc
    if not isinstance(extra_head_kwargs, dict):
        raise ValueError("--head-kwargs-json must decode to a JSON object.")

    return build_expression_head(
        args.head,
        sequence_embedding_dim=sequence_embedding_dim,
        host_embedding_dim=host_embedding_dim,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
        output_dim=1,
        activation=args.activation,
        fusion_dim=args.fusion_dim,
        film_hidden_dims=args.film_hidden_dims,
        film_use_layer_norm=not args.no_film_layer_norm,
        film_gamma_scale=args.film_gamma_scale,
        film_include_host_skip=args.film_include_host_skip,
        film_identity_init=not args.no_film_identity_init,
        **extra_head_kwargs,
    )


def run_training(args: Any) -> None:
    """Run one complete train/validate/test job from an argparse namespace."""

    resolve_paths(args)
    enforce_host_specific_baseline(args)
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets, reports, df = make_datasets(args)
    reports_dict = {split: report.__dict__ for split, report in reports.items()}
    save_json(run_output_path(args, "split_build_report.json"), reports_dict)

    train_targets = np.asarray([record.target for record in datasets["train"].records], dtype=np.float64)
    if args.standardize_target:
        target_mean = float(np.mean(train_targets))
        target_std = float(np.std(train_targets))
        if target_std == 0.0 or not math.isfinite(target_std):
            target_std = 1.0
    else:
        target_mean = 0.0
        target_std = 1.0

    train_loader = make_loader(datasets["train"], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, device=device)
    val_loader = make_loader(datasets["val"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, device=device)
    test_loader = make_loader(datasets["test"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, device=device)

    sequence_embedding_dim = datasets["train"].sequence_embedding_dim
    host_embedding_dim = datasets["train"].host_embedding_dim
    model = build_model_from_args(args, sequence_embedding_dim=sequence_embedding_dim, host_embedding_dim=host_embedding_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.loss == "mse":
        criterion = nn.MSELoss(reduction="none")
    else:
        criterion = nn.HuberLoss(delta=args.huber_delta, reduction="none")

    autocast_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=autocast_enabled) if device.type == "cuda" else None
    use_weights = args.weight_col is not None

    config = {
        "run": {
            "name": args.run_name,
            "output_dir": str(args.output_dir),
            "artifact_prefix": f"{args.run_name}_",
        },
        "args": to_serializable(vars(args)),
        "head": args.head,
        "head_config": expression_head_config_dict(model),
        "sequence_embedding_dim": sequence_embedding_dim,
        "host_embedding_dim": host_embedding_dim,
        "conditions": {
            "requested": args.conditions,
            "table_observed": sorted(df[args.condition_col].astype(str).unique().tolist()),
            "train_observed": datasets["train"].conditions,
            "val_observed": datasets["val"].conditions,
            "test_observed": datasets["test"].conditions,
            "train_counts": datasets["train"].condition_counts,
            "val_counts": datasets["val"].condition_counts,
            "test_counts": datasets["test"].condition_counts,
        },
        "target": {
            "column": args.target_col,
            "standardized": bool(args.standardize_target),
            "mean": target_mean,
            "std": target_std,
        },
        "split_build_report": reports_dict,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "device": str(device),
    }
    save_json(run_output_path(args, "config.json"), config)
    print(json.dumps({"config": config}, indent=2, default=str))

    best_val_mse = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []
    best_path = run_output_path(args, f"best_{args.head}_head.pt")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            device=device,
            use_weights=use_weights,
            target_mean=target_mean,
            target_std=target_std,
            autocast_enabled=autocast_enabled,
            grad_clip_norm=args.grad_clip_norm,
        )
        val_metrics, _ = evaluate(
            model,
            val_loader,
            criterion=criterion,
            device=device,
            use_weights=use_weights,
            target_mean=target_mean,
            target_std=target_std,
            autocast_enabled=autocast_enabled,
            return_predictions=False,
        )

        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        save_json(run_output_path(args, "history.json"), history)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.6g} train_rmse={train_metrics['rmse']:.6g} "
            f"val_loss={val_metrics['loss']:.6g} val_rmse={val_metrics['rmse']:.6g} "
            f"val_pearson={val_metrics['pearson']:.6g}"
        )

        val_mse = float(val_metrics["mse"])
        improved = val_mse < (best_val_mse - args.min_delta)
        if improved:
            best_val_mse = val_mse
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "head": args.head,
                    "model_class": type(model).__name__,
                    "head_config": expression_head_config_dict(model),
                    "sequence_embedding_dim": sequence_embedding_dim,
                    "host_embedding_dim": host_embedding_dim,
                    "target": config["target"],
                    "args": to_serializable(vars(args)),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                print(f"early stopping at epoch {epoch}; best_epoch={best_epoch}")
                break

    if not best_path.exists():
        raise RuntimeError("Training finished without writing a best checkpoint.")

    try:
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    final_metrics: dict[str, dict[str, float]] = {}
    for split, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        metrics, predictions = evaluate(
            model,
            loader,
            criterion=criterion,
            device=device,
            use_weights=use_weights,
            target_mean=target_mean,
            target_std=target_std,
            autocast_enabled=autocast_enabled,
            return_predictions=True,
        )
        final_metrics[split] = metrics
        assert predictions is not None
        predictions = predictions.rename(columns={"y_true": f"y_true_{args.target_col}", "y_pred": f"y_pred_{args.target_col}"})
        predictions.to_csv(run_output_path(args, f"{split}_predictions.tsv"), sep="\t", index=False)

    save_json(run_output_path(args, "metrics.json"), final_metrics)
    checkpoint["final_metrics"] = final_metrics
    torch.save(checkpoint, best_path)

    print("best checkpoint:", best_path)
    print(json.dumps({"best_epoch": best_epoch, "metrics": final_metrics}, indent=2, allow_nan=True))


__all__ = [
    "build_model_from_args",
    "enforce_host_specific_baseline",
    "evaluate",
    "loss_from_batch",
    "make_datasets",
    "make_loader",
    "resolve_paths",
    "run_output_path",
    "run_training",
    "safe_run_name",
    "save_json",
    "set_seed",
    "to_serializable",
    "train_one_epoch",
]
