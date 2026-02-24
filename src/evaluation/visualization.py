"""
WSI Attention Heatmap Visualization for Thyroid TERT Mutation Prediction

- Reads attention scores from attention_scores/attention_scores_fold{N}.json
- Reads CV metrics from results_cv_summary_optimal.json (fallback: results_cv_summary.json)
- Supports TERT dataset structure: embedding/{C228T|C250T|Wild}/json/
- Generates heatmaps for TEST SET predictions (correct/incorrect cases)

Usage:
    from evaluation.visualization import generate_attention_heatmaps_from_results

    generate_attention_heatmaps_from_results(
        results_dir="/path/to/outputs/thyroid_tert_v0.1.1",
        embedding_base_dir="/path/to/Thyroid_TERT_dataset/embedding",
        save_dir="/path/to/outputs/thyroid_tert_v0.1.1/heatmaps",
        fold_num="best",
        n_correct=3,
        n_incorrect=3
    )
"""

import os
import json
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import openslide
except ImportError:
    openslide = None

from scipy.ndimage import gaussian_filter

# Matplotlib settings (avoid permission issues)
os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache_wsi"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# =========================
# Helpers
# =========================
def create_attention_heatmap_colormap():
    """Attention score colormap (blue -> green -> yellow -> red)."""
    colors = ["#2E3192", "#1BFFFF", "#00FF00", "#FFFF00", "#FF0000"]
    return LinearSegmentedColormap.from_list("attention", colors, N=256)


def load_json_metadata(json_path: Path) -> Dict:
    """Load JSON metadata."""
    with open(json_path, "r") as f:
        return json.load(f)


def get_class_from_npy_path(npy_path: str) -> Optional[str]:
    """
    Extract class name (C228T, C250T, Wild) from npy file path.

    Path structure: .../embedding/{C228T|C250T|Wild}/npy/{sample_id}.npy
    """
    parts = Path(npy_path).parts
    try:
        npy_idx = parts.index('npy')
        class_name = parts[npy_idx - 1]
        if class_name in ['C228T', 'C250T', 'Wild']:
            return class_name
    except (ValueError, IndexError):
        pass
    return None


def find_json_metadata(
    wsi_name: str,
    embedding_base_dir: Path,
    npy_path: Optional[str] = None
) -> Optional[Path]:
    """
    Locate JSON metadata file for a given WSI.

    TERT dataset structure: embedding/{C228T|C250T|Wild}/json/{sample_id}.json

    Args:
        wsi_name: WSI sample ID (e.g., TC_04_8001)
        embedding_base_dir: Base directory for embeddings
        npy_path: Optional npy file path to extract class from

    Returns:
        Path to JSON metadata or None if not found.
    """
    wsi_name = wsi_name.replace(".npy", "").replace(".pt", "")

    # If npy_path is provided, extract class from it
    if npy_path:
        class_name = get_class_from_npy_path(npy_path)
        if class_name:
            json_path = embedding_base_dir / class_name / "json" / f"{wsi_name}.json"
            if json_path.exists():
                return json_path

    # Search in all class directories
    for class_name in ['C228T', 'C250T', 'Wild']:
        json_path = embedding_base_dir / class_name / "json" / f"{wsi_name}.json"
        if json_path.exists():
            return json_path

    return None


def extract_coordinates_from_json(metadata: Dict) -> Tuple[Dict, Tuple[int, int]]:
    """
    Extract patch coordinates from JSON metadata.

    TERT JSON structure:
    {
        "slide_id": "TC_04_8001",
        "num_tiles": 27465,
        "patch_coords": [{"name": "...", "x": 0, "y": 100352}, ...]
    }

    Returns:
        coords_dict (dict): {patch_idx: (row, col)}
        grid_shape (tuple): (n_rows, n_cols)
    """
    # Extract tiles from different possible structures
    if isinstance(metadata, dict):
        if "patch_coords" in metadata:
            tiles = metadata["patch_coords"]
        elif "tiles" in metadata:
            tiles = metadata["tiles"]
        elif "coords" in metadata:
            tiles = metadata["coords"]
        else:
            # Check if values are coordinate dicts
            if all(isinstance(v, dict) and "x" in v and "y" in v for v in metadata.values()):
                tiles = list(metadata.values())
            else:
                raise KeyError(f"Cannot find coordinate keys in JSON. Available keys: {list(metadata.keys())}")
    elif isinstance(metadata, list):
        tiles = metadata
    else:
        raise TypeError(f"Unsupported metadata type: {type(metadata)}")

    # Extract x, y coordinates
    x_coords = [tile["x"] for tile in tiles if "x" in tile]
    y_coords = [tile["y"] for tile in tiles if "y" in tile]

    if len(x_coords) == 0 or len(y_coords) == 0:
        raise ValueError("No coordinate data found in metadata.")

    # Create grid mapping
    x_coords_unique = sorted(set(x_coords))
    y_coords_unique = sorted(set(y_coords))
    n_cols = len(x_coords_unique)
    n_rows = len(y_coords_unique)
    grid_shape = (n_rows, n_cols)

    x_to_col = {x: i for i, x in enumerate(x_coords_unique)}
    y_to_row = {y: i for i, y in enumerate(y_coords_unique)}

    coords_dict = {}
    for idx, tile in enumerate(tiles):
        if "x" not in tile or "y" not in tile:
            continue
        row = y_to_row.get(tile["y"], 0)
        col = x_to_col.get(tile["x"], 0)
        coords_dict[idx] = (row, col)

    return coords_dict, grid_shape


def extract_tiles_from_metadata(metadata: Dict) -> List[Dict]:
    """Extract tile list from metadata with flexible key handling."""
    if isinstance(metadata, dict):
        if "patch_coords" in metadata:
            tiles = metadata["patch_coords"]
        elif "tiles" in metadata:
            tiles = metadata["tiles"]
        elif "coords" in metadata:
            tiles = metadata["coords"]
        else:
            if all(isinstance(v, dict) and "x" in v and "y" in v for v in metadata.values()):
                tiles = list(metadata.values())
            else:
                raise KeyError(f"Cannot find coordinate keys in JSON. Available keys: {list(metadata.keys())}")
    elif isinstance(metadata, list):
        tiles = metadata
    else:
        raise TypeError(f"Unsupported metadata type: {type(metadata)}")
    return tiles


def _estimate_patch_size(sorted_coords: List[int], default_size: int = 512) -> int:
    """Estimate patch size from coordinate step."""
    if len(sorted_coords) < 2:
        return default_size
    diffs = np.diff(sorted_coords)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return default_size
    return int(np.median(diffs))


def _build_svs_path(wsi_name: str, svs_base_dir: Optional[Path]) -> Optional[Path]:
    if svs_base_dir is None:
        return None
    path = svs_base_dir / f"{wsi_name}.svs"
    return path if path.exists() else None


def save_heatmap_thumbnail_overlay(
    wsi_name: str,
    heatmap: np.ndarray,
    metadata: Dict,
    save_dir: Path,
    case_type: str,
    case_idx: int,
    cmap,
    svs_base_dir: Optional[Path],
    thumbnail_max_side: int = 2048,
    overlay_alpha: float = 0.85,
    overlay_gamma: float = 0.60,
    overlay_alpha_floor: float = 0.15,
) -> Optional[Path]:
    """Save thumbnail overlay image if OpenSlide and SVS are available."""
    if openslide is None:
        print("  [i] Skipping overlay: openslide-python not installed")
        return None

    svs_path = _build_svs_path(wsi_name, svs_base_dir)
    if svs_path is None:
        print(f"  [i] Skipping overlay: SVS not found for {wsi_name}")
        return None

    tiles = extract_tiles_from_metadata(metadata)
    x_coords = sorted(set([t["x"] for t in tiles if "x" in t]))
    y_coords = sorted(set([t["y"] for t in tiles if "y" in t]))
    if len(x_coords) == 0 or len(y_coords) == 0:
        print(f"  [i] Skipping overlay: invalid coordinates for {wsi_name}")
        return None

    patch_w = _estimate_patch_size(x_coords, default_size=512)
    patch_h = _estimate_patch_size(y_coords, default_size=512)

    min_x, max_x = x_coords[0], x_coords[-1]
    min_y, max_y = y_coords[0], y_coords[-1]
    bbox_x0 = int(min_x)
    bbox_y0 = int(min_y)
    bbox_x1 = int(max_x + patch_w)
    bbox_y1 = int(max_y + patch_h)

    slide = openslide.OpenSlide(str(svs_path))
    slide_w, slide_h = slide.dimensions
    scale = min(float(thumbnail_max_side) / max(slide_w, slide_h), 1.0)
    thumb_w = max(1, int(slide_w * scale))
    thumb_h = max(1, int(slide_h * scale))
    thumb = slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB")
    slide.close()

    thumb_np = np.array(thumb).astype(np.float32) / 255.0

    x0 = max(0, min(thumb_w - 1, int(round(bbox_x0 * scale))))
    y0 = max(0, min(thumb_h - 1, int(round(bbox_y0 * scale))))
    x1 = max(x0 + 1, min(thumb_w, int(round(bbox_x1 * scale))))
    y1 = max(y0 + 1, min(thumb_h, int(round(bbox_y1 * scale))))

    if cv2 is not None:
        heatmap_resized = cv2.resize(
            heatmap.astype(np.float32),
            (x1 - x0, y1 - y0),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        from scipy.ndimage import zoom
        zoom_y = (y1 - y0) / max(heatmap.shape[0], 1)
        zoom_x = (x1 - x0) / max(heatmap.shape[1], 1)
        heatmap_resized = zoom(heatmap.astype(np.float32), (zoom_y, zoom_x), order=1)
        heatmap_resized = heatmap_resized[: (y1 - y0), : (x1 - x0)]

    score = np.clip(heatmap_resized, 0.0, 1.0)
    score_enhanced = np.power(score, overlay_gamma)
    heat_rgb = cmap(score_enhanced)[..., :3]

    alpha_mask = np.where(
        score > 0,
        (overlay_alpha_floor + (1.0 - overlay_alpha_floor) * score_enhanced) * overlay_alpha,
        0.0,
    )
    alpha_mask = np.clip(alpha_mask, 0.0, 1.0)[..., None]

    region = thumb_np[y0:y1, x0:x1, :]
    blended = region * (1.0 - alpha_mask) + heat_rgb * alpha_mask
    thumb_np[y0:y1, x0:x1, :] = blended

    overlay_img = np.clip(thumb_np * 255.0, 0, 255).astype(np.uint8)
    overlay_path = save_dir / f"{case_type}_{case_idx:02d}_{wsi_name}_overlay.png"
    plt.figure(figsize=(12, 8))
    plt.imshow(overlay_img)
    plt.title(f"{wsi_name} - Heatmap Overlay", fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(overlay_path, dpi=200, bbox_inches="tight")
    plt.close()

    return overlay_path


def create_attention_heatmap(
    attention_scores: np.ndarray,
    coords_dict: Dict,
    grid_shape: Tuple[int, int],
    patch_indices: Optional[range] = None,
    interpolation: str = "gaussian"
) -> np.ndarray:
    """Convert attention scores to spatial heatmap."""
    if cv2 is None and interpolation == "gaussian":
        raise ImportError("opencv-python is required for heatmap generation. Install with: pip install opencv-python")

    n_rows, n_cols = grid_shape
    heatmap = np.zeros((n_rows, n_cols))
    count_map = np.zeros((n_rows, n_cols))

    if patch_indices is None:
        patch_indices = range(len(attention_scores))

    for patch_idx in patch_indices:
        if patch_idx >= len(attention_scores):
            continue
        score = attention_scores[patch_idx]
        if patch_idx in coords_dict:
            row, col = coords_dict[patch_idx]
            if 0 <= row < n_rows and 0 <= col < n_cols:
                heatmap[row, col] += score
                count_map[row, col] += 1

    # Average overlapping scores
    mask = count_map > 0
    heatmap[mask] /= count_map[mask]

    # Interpolation
    if interpolation == "gaussian":
        if np.sum(~mask) > 0:
            heatmap_filled = cv2.inpaint(
                (heatmap * 255).astype(np.uint8),
                (~mask).astype(np.uint8),
                inpaintRadius=3,
                flags=cv2.INPAINT_TELEA,
            ) / 255.0
        else:
            heatmap_filled = heatmap
        heatmap = gaussian_filter(heatmap_filled, sigma=1.0)
    elif interpolation == "bilinear":
        from scipy.ndimage import zoom
        scale = 4
        heatmap_upscaled = zoom(heatmap, scale, order=1)
        heatmap = zoom(heatmap_upscaled, 1 / scale, order=1)

    return heatmap


def get_label_display_name(label: int, mutation_type: Optional[str] = None) -> str:
    """Get display name for TERT mutation label."""
    if label == 1:
        if mutation_type:
            return f"Mutant ({mutation_type})"
        return "Mutant"
    return "Wild"


def visualize_single_heatmap(
    wsi_name: str,
    wsi_data: Dict,
    embedding_base_dir: Path,
    save_dir: Path,
    cmap,
    npy_path: Optional[str] = None,
    case_type: str = "correct",
    case_idx: int = 1,
    interpolation: str = "gaussian",
    dpi: int = 200,
    svs_base_dir: Optional[Path] = None,
    thumbnail_max_side: int = 2048,
    overlay_alpha: float = 0.85,
    overlay_gamma: float = 0.60,
    overlay_alpha_floor: float = 0.15,
):
    """Visualize a single WSI attention heatmap."""
    attention_scores = np.array(wsi_data["scores"])
    n_patches = wsi_data["n_patches"]
    model_type = str(wsi_data.get("model_type", "")).lower()
    score_type = str(wsi_data.get("score_type", "attention")).lower()
    true_label = wsi_data.get("true_label")
    pred_label = wsi_data.get("predicted_label")
    pred_prob = wsi_data.get("pred_prob")

    # Get mutation type from path
    mutation_type = get_class_from_npy_path(npy_path) if npy_path else None

    print(f"\n[{case_idx}] {wsi_name}")
    print(f"  Type: {'Correct' if case_type == 'correct' else 'Incorrect'}")
    if true_label is not None:
        print(f"  True Label: {get_label_display_name(true_label, mutation_type if true_label == 1 else None)}")
    if pred_label is not None and pred_prob is not None:
        print(f"  Predicted: {get_label_display_name(pred_label)} (prob={pred_prob:.3f})")
    print(f"  Patches: {n_patches}")
    print(f"  Score Type: {score_type}")
    print(f"  Score Range (Raw): [{attention_scores.min():.6f}, {attention_scores.max():.6f}]")

    # Normalize attention scores
    if model_type == "transmil" and score_type == "transmil_b_attribution":
        # Attribution scores are often sparse/small; use robust percentile scaling
        # for visualization only (does not affect model outputs).
        pos_scores = attention_scores[attention_scores > 0]
        if pos_scores.size > 0:
            lo = np.percentile(pos_scores, 5)
            hi = np.percentile(pos_scores, 99)
            denom = max(hi - lo, 1e-8)
            attention_scores_normalized = np.clip((attention_scores - lo) / denom, 0.0, 1.0)
            print(f"  Normalization: robust percentile (p5={lo:.6g}, p99={hi:.6g})")
        else:
            attention_scores_normalized = np.zeros_like(attention_scores)
            print("  Normalization: all-zero attribution scores")
    # For other TransMIL score types, keep raw scale.
    elif model_type == "transmil":
        attention_scores_normalized = attention_scores
        print(f"  Normalization: skipped (model_type=transmil)")
    else:
        if attention_scores.max() > attention_scores.min():
            attention_scores_normalized = (attention_scores - attention_scores.min()) / (
                attention_scores.max() - attention_scores.min()
            )
        else:
            attention_scores_normalized = attention_scores

    print(f"  Score Range (Normalized): [{attention_scores_normalized.min():.3f}, {attention_scores_normalized.max():.3f}]")

    # Find JSON metadata
    json_path = find_json_metadata(wsi_name, embedding_base_dir, npy_path)
    if json_path is None:
        print(f"  [!] Skipping: Cannot find JSON metadata for {wsi_name}")
        return

    metadata = load_json_metadata(json_path)
    print(f"  Loaded: {json_path.name}")

    # Extract coordinates and create heatmap
    coords_dict, grid_shape = extract_coordinates_from_json(metadata)
    print(f"  Grid Shape: {grid_shape[0]} rows x {grid_shape[1]} cols")

    heatmap = create_attention_heatmap(
        attention_scores_normalized,
        coords_dict,
        grid_shape,
        patch_indices=range(n_patches),
        interpolation=interpolation
    )

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    im = ax.imshow(heatmap, cmap=cmap, aspect="auto", interpolation="bilinear")

    # Title
    title_parts = [f"{wsi_name}"]
    title_parts.append("Correct Prediction" if case_type.startswith("correct") else "Incorrect Prediction")
    if true_label is not None and pred_label is not None:
        true_str = get_label_display_name(true_label, mutation_type if true_label == 1 else None)
        pred_str = get_label_display_name(pred_label)
        title_parts.append(f"True: {true_str} | Pred: {pred_str} ({pred_prob:.3f})")

    ax.set_title("\n".join(title_parts), fontsize=14, fontweight="bold", pad=20)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if model_type == "transmil" and score_type == "transmil_b_attribution":
        cbar_label = "TransMIL-B Attribution Score"
    elif model_type == "transmil":
        cbar_label = "TransMIL Score"
    else:
        cbar_label = "Attention Score (Normalized)\n[0.000, 1.000]"
    cbar.set_label(cbar_label, fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=9)

    ax.set_xlabel("Column Index", fontsize=11)
    ax.set_ylabel("Row Index", fontsize=11)
    ax.grid(False)

    # Save
    filename = f"{case_type}_{case_idx:02d}_{wsi_name}_heatmap.png"
    save_path = save_dir / filename
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")

    overlay_path = save_heatmap_thumbnail_overlay(
        wsi_name=wsi_name,
        heatmap=heatmap,
        metadata=metadata,
        save_dir=save_dir,
        case_type=case_type,
        case_idx=case_idx,
        cmap=cmap,
        svs_base_dir=svs_base_dir,
        thumbnail_max_side=thumbnail_max_side,
        overlay_alpha=overlay_alpha,
        overlay_gamma=overlay_gamma,
        overlay_alpha_floor=overlay_alpha_floor,
    )
    if overlay_path is not None:
        print(f"  Saved: {overlay_path.name}")

    gc.collect()


def generate_attention_heatmaps_from_results(
    results_dir: str,
    embedding_base_dir: str,
    save_dir: str,
    fold_num: str = "best",
    n_per_class: int = 5,
    interpolation: str = "gaussian",
    dpi: int = 200,
    svs_base_dir: str = "/path/to/Thyroid_TERT_dataset/thyroid",
    thumbnail_max_side: int = 2048,
    overlay_alpha: float = 0.85,
    overlay_gamma: float = 0.60,
    overlay_alpha_floor: float = 0.15,
):
    """
    Generate attention heatmaps from TERT training results.

    서브타입별(C228T, C250T, Wild) 각 n_per_class장씩 confidence 높은 순으로
    correct prediction에 대해 heatmap + overlay 생성.

    Args:
        results_dir: Directory containing training results
        embedding_base_dir: Base directory for embeddings
        save_dir: Directory to save heatmaps
        fold_num: Fold number to use ("best" for best AUC fold, or specific fold number)
        n_per_class: Number of correct predictions per subtype (C228T, C250T, Wild)
        interpolation: Interpolation method ("gaussian" or "bilinear")
        dpi: DPI for saved images
    """
    print(f"\n{'='*80}")
    print(f"TERT Attention Heatmap Generation")
    print(f"{'='*80}")
    print(f"Results dir: {results_dir}")
    print(f"Embedding dir: {embedding_base_dir}")
    print(f"{'='*80}\n")

    results_dir = Path(results_dir)
    embedding_base_dir = Path(embedding_base_dir)
    svs_base_dir_path = Path(svs_base_dir) if svs_base_dir else None

    # 1) Load CV summary
    cv_summary_path = results_dir / "results_cv_summary_optimal.json"
    if not cv_summary_path.exists():
        cv_summary_path = results_dir / "results_cv_summary.json"
    if not cv_summary_path.exists():
        print(f"[!] CV summary not found under {results_dir}")
        return

    with open(cv_summary_path, "r") as f:
        cv_summary = json.load(f)
    print(f"[+] Loaded CV summary from: {cv_summary_path.name}")

    # 2) Select fold
    if fold_num == "best":
        best_fold = max(cv_summary["folds"], key=lambda x: x["test_metrics"]["auc"])
        fold_idx = best_fold["fold"]
        print(f"[+] Selected best fold: Fold {fold_idx} (AUC: {best_fold['test_metrics']['auc']:.4f})")
    else:
        fold_idx = int(fold_num)
        best_fold = next((f for f in cv_summary["folds"] if f["fold"] == fold_idx), None)
        if not best_fold:
            print(f"[!] Fold {fold_idx} not found in results")
            return
        print(f"[+] Selected fold: Fold {fold_idx}")

    # 3) Load attention scores
    attention_dir = results_dir / "attention_scores"
    attention_path = attention_dir / f"attention_scores_fold{fold_idx}.json"
    if not attention_path.exists():
        print(f"[!] Attention scores not found: {attention_path}")
        print("    Make sure you ran training with --generate_plots flag")
        return

    with open(attention_path, "r") as f:
        attention_data = json.load(f)
    attention_scores_dict = attention_data["attention_scores"]
    default_model_type = str(
        attention_data.get("model_type", cv_summary.get("model_type", ""))
    ).lower()
    if not default_model_type and "transmil" in str(results_dir).lower():
        default_model_type = "transmil"
    print(f"[+] Loaded attention scores for {len(attention_scores_dict)} WSIs")

    # 4) Build npy_path mapping from CV splits
    cv_split_path = results_dir.parent.parent / "config" / "cv_splits_tert_5fold_seed42.json"
    npy_path_mapping = {}
    if cv_split_path.exists():
        with open(cv_split_path, "r") as f:
            cv_splits = json.load(f)
        for fold_data in cv_splits["folds"]:
            for split in ["train_wsis", "val_wsis", "test_wsis"]:
                for path in fold_data.get(split, []):
                    sample_id = Path(path).stem
                    npy_path_mapping[sample_id] = path

    # 5) Prepare prediction results
    prediction_results = {}
    for wsi_name, attn_data in attention_scores_dict.items():
        if default_model_type and "model_type" not in attn_data:
            attn_data["model_type"] = default_model_type
        if "true_label" in attn_data and "predicted_label" in attn_data:
            prediction_results[wsi_name] = {
                "label": attn_data["true_label"],
                "prediction": attn_data["predicted_label"],
                "probability": attn_data.get("pred_prob", 0.5),
                "is_correct": attn_data["true_label"] == attn_data["predicted_label"],
                "npy_path": npy_path_mapping.get(wsi_name),
            }
    print(f"[+] Prepared prediction results for {len(prediction_results)} WSIs")

    # 6) Select cases — split correct into C228T / C250T / Wild
    subtype_cases: Dict[str, List] = {"C228T": [], "C250T": [], "Wild": []}
    for wsi_name, pred_info in prediction_results.items():
        if not pred_info["is_correct"]:
            continue
        npy_path = pred_info.get("npy_path") or ""
        subtype = get_class_from_npy_path(npy_path)
        if subtype and subtype in subtype_cases:
            entry = (wsi_name, attention_scores_dict[wsi_name], npy_path)
            subtype_cases[subtype].append(entry)

    # Sort by confidence (distance from 0.5) and take top n_per_class
    for subtype in subtype_cases:
        subtype_cases[subtype].sort(
            key=lambda x: abs(x[1].get("pred_prob", 0.5) - 0.5), reverse=True
        )
        subtype_cases[subtype] = subtype_cases[subtype][:n_per_class]

    print(f"\nSelected Cases (Fold {fold_idx}):")
    for subtype in ["C228T", "C250T", "Wild"]:
        print(f"  Correct {subtype}: {len(subtype_cases[subtype])}/{n_per_class}")

    # 7) Create save directory
    save_dir = Path(save_dir) / f"fold_{fold_idx}_attention_heatmaps"
    save_dir.mkdir(parents=True, exist_ok=True)
    cmap = create_attention_heatmap_colormap()

    # 8) Generate heatmaps per subtype
    for subtype in ["C228T", "C250T", "Wild"]:
        cases = subtype_cases[subtype]
        print(f"\n{'-'*80}")
        print(f"Processing Correct {subtype} Predictions ({len(cases)})")
        print(f"{'-'*80}")
        for idx, (wsi_name, wsi_data, npy_path) in enumerate(cases, 1):
            visualize_single_heatmap(
                wsi_name,
                wsi_data,
                embedding_base_dir,
                save_dir,
                cmap,
                npy_path=npy_path,
                case_type=f"correct_{subtype}",
                case_idx=idx,
                interpolation=interpolation,
                dpi=dpi,
                svs_base_dir=svs_base_dir_path,
                thumbnail_max_side=thumbnail_max_side,
                overlay_alpha=overlay_alpha,
                overlay_gamma=overlay_gamma,
                overlay_alpha_floor=overlay_alpha_floor,
            )

    print(f"\n{'='*80}")
    print(f"[+] Attention heatmap generation complete!")
    print(f"[+] Saved to: {save_dir}")
    print(f"{'='*80}\n")


# =========================
# CLI Entry Point
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate TERT Attention Heatmaps")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing training results")
    parser.add_argument("--embedding_base_dir", type=str,
                        default="/path/to/Thyroid_TERT_dataset/embedding",
                        help="Base directory for embeddings")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save heatmaps (default: results_dir/heatmaps)")
    parser.add_argument("--fold", type=str, default="best",
                        help="Fold number or 'best' for best AUC fold")
    parser.add_argument("--n_per_class", type=int, default=5,
                        help="Number of correct predictions per subtype (C228T, C250T, Wild)")
    parser.add_argument("--interpolation", type=str, default="gaussian",
                        choices=["gaussian", "bilinear"],
                        help="Interpolation method")
    parser.add_argument("--dpi", type=int, default=200,
                        help="DPI for saved images")
    parser.add_argument("--svs_base_dir", type=str,
                        default="/path/to/Thyroid_TERT_dataset/thyroid",
                        help="Directory containing .svs files for thumbnail overlay")
    parser.add_argument("--thumbnail_max_side", type=int, default=2048,
                        help="Max side length for overlay thumbnail image")
    parser.add_argument("--overlay_alpha", type=float, default=0.85,
                        help="Global alpha scale for thumbnail overlay")
    parser.add_argument("--overlay_gamma", type=float, default=0.60,
                        help="Gamma for overlay contrast (<1 strengthens mid-range)")
    parser.add_argument("--overlay_alpha_floor", type=float, default=0.15,
                        help="Minimum alpha for non-zero heatmap region")

    args = parser.parse_args()

    save_dir = args.save_dir or str(Path(args.results_dir) / "heatmaps")

    generate_attention_heatmaps_from_results(
        results_dir=args.results_dir,
        embedding_base_dir=args.embedding_base_dir,
        save_dir=save_dir,
        fold_num=args.fold,
        n_per_class=args.n_per_class,
        interpolation=args.interpolation,
        dpi=args.dpi,
        svs_base_dir=args.svs_base_dir,
        thumbnail_max_side=args.thumbnail_max_side,
        overlay_alpha=args.overlay_alpha,
        overlay_gamma=args.overlay_gamma,
        overlay_alpha_floor=args.overlay_alpha_floor,
    )
