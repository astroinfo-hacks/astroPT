import argparse
import csv
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import umap
except ImportError as exc:
    raise SystemExit("The 'umap-learn' package is required. Install it with 'pip install umap-learn'.") from exc

try:
    from sklearn.ensemble import IsolationForest
except ImportError as exc:
    raise SystemExit("scikit-learn is required. Install it with 'pip install scikit-learn'.") from exc


def load_records(path: Path, idx_path: Path = None) -> list[dict]:
    try:
        # .pt file case
        data = torch.load(path, map_location="cpu")
    except:
        # .npy file case
        data = np.load(path)
        idxs = np.load(idx_path)
        
    if isinstance(data, (list, np.ndarray)):
        if idx_path is not None:
            return data, idx_path
        else:
            return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported embeddings format: {type(data)}")


def stack_embeddings(records: Sequence[dict], key: str) -> np.ndarray:
    vectors = []
    for rec in records:
        #tensor = rec.get(key)
        tensor=rec
        if tensor is None:
            continue
        if isinstance(tensor, torch.Tensor):
            vectors.append(tensor.detach().cpu().numpy())
        else:
            vectors.append(np.asarray(tensor))
    if not vectors:
        raise ValueError(f"No embeddings found for key '{key}'")
    return np.stack(vectors, axis=0)


def run_isolation_forest(embeddings: np.ndarray, contamination: float, random_state: int) -> np.ndarray:
    model = IsolationForest(contamination=contamination, random_state=random_state)
    labels = model.fit_predict(embeddings)
    return labels == -1  # True for outliers


def compute_umap(embeddings: np.ndarray, random_state: int) -> np.ndarray:
    reducer = umap.UMAP(random_state=random_state)
    return reducer.fit_transform(embeddings)


def plot_pair_umaps(
    coords: np.ndarray,
    mask_primary: np.ndarray,
    titles: tuple[str, str],
    labels: tuple[str, str],
    save_path: Path,
    mask_secondary: np.ndarray = None,
) -> None:
    plt.figure(figsize=(12, 6))

    title=titles
    label=labels
    highlight_mask=mask_primary
    print(highlight_mask.shape, highlight_mask)
    plt.scatter(
        coords[~highlight_mask, 0],
        coords[~highlight_mask, 1],
        s=8,
        color="lightgray",
        alpha=0.5,
        label="Inliers",
    )
    if highlight_mask.any():
        plt.scatter(
            coords[highlight_mask, 0],
            coords[highlight_mask, 1],
            s=24,
            color="crimson",
            edgecolors="black",
            linewidths=0.5,
            label=label,
        )
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.grid(True, alpha=0.2)
    plt.legend(loc="best")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=220)
    plt.close()


def save_umap_csv(
    path: Path,
    object_ids: Sequence[str],
    coords_euc: np.ndarray,
    coords_euc_desi: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "object_id",
            "umap_euc_x",
            "umap_euc_y",
            "umap_euc_desi_x",
            "umap_euc_desi_y",
        ])
        for oid, (xh, yh), (xsd, ysd) in zip(object_ids, coords_euc, coords_euc_desi):
            writer.writerow([oid, f"{xh:.6f}", f"{yh:.6f}", f"{xsd:.6f}", f"{ysd:.6f}"])


def save_outlier_ids(path: Path, object_ids: Sequence[str], mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pawth.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["object_id"])
        for oid in np.array(object_ids)[mask]:
            writer.writerow([oid])


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Detect embedding outliers with Isolation Forest and visualize them on UMAP.")
    parser.add_argument("--input", required=True, help="Path to embeddings file")
    parser.add_argument("--indices", required=True, help="Path to indices file")
    parser.add_argument("--figure", required=True, help="Path to save Euclid UMAP composite figure")
    #parser.add_argument("--figure-euc-desi", required=True, help="Path to save euc+DESI UMAP composite figure")
    #parser.add_argument("--umap-csv", required=True, help="CSV path to store UMAP coordinates")
    parser.add_argument("--outliers", required=True, help="CSV path to store Euclid outlier object IDs")
    #parser.add_argument("--outliers-euc-desi", required=True, help="CSV path to store euc+DESI outlier object IDs")
    parser.add_argument("--contamination", type=float, default=0.02, help="Isolation Forest contamination fraction (default: 0.02)")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility")
    args = parser.parse_args(argv)

    records, object_ids = load_records(Path(args.input),Path(args.indices))

    emb_euc = stack_embeddings(records, "embedding_euc")
    #emb_euc_desi = stack_embeddings(records, "embedding_euc_desi")

    outliers_euc = run_isolation_forest(emb_euc, args.contamination, args.random_state)
    print("number of outliers",outliers_euc.sum())
    #outliers_euc_desi = run_isolation_forest(emb_euc_desi, args.contamination, args.random_state)

    coords_euc = compute_umap(emb_euc, random_state=args.random_state)
    print(coords_euc.shape)
    #coords_euc_desi = compute_umap(emb_euc_desi, random_state=args.random_state)

    plot_pair_umaps(
        coords_euc,
        mask_primary=outliers_euc,
        mask_secondary=None,
        titles=("euclid UMAP – euclid outliers"),
        labels=("euclid outliers"),
        save_path=Path(args.figure),
    )
    """
    plot_pair_umaps(
        coords_euc_desi,
        mask_primary=outliers_euc_desi,
        mask_secondary=outliers_euc,
        titles=("euc+DESI UMAP – euc+DESI outliers", "euc+DESI UMAP – euc outliers"),
        labels=("euc+DESI outliers", "euc outliers"),
        save_path=Path(args.figure_euc_desi),
    )
    """

    #save_umap_csv(Path(args.umap_csv), object_ids, coords_euc, coords_euc_desi)
    save_outlier_ids(Path(args.outliers), object_ids, outliers_euc)
    #save_outlier_ids(Path(args.outliers_euc_desi), object_ids, outliers_euc_desi)

    #print(f"Detected {outliers_euc.sum()} euc outliers and {outliers_euc_desi.sum()} euc+DESI outliers.")
    #print(f"UMAP coordinates saved to {args.umap_csv}")
    #print(f"Outlier ID lists saved to {args.outliers_euc} and {args.outliers_euc_desi}")


if __name__ == "__main__":
    main()
