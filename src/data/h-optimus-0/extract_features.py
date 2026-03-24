"""
H-Optimus-0 Feature Extraction for Thyroid TERT Dataset (DDP)

Usage:
    torchrun --nproc_per_node=4 --master_port=29502 extract_features.py \
        --tile_dir /path/to/dataset/40x_patch/C228T \
        --out_dir  /path/to/dataset/h_optimus_embeddings/C228T \
        --batch_size 448

클래스별로 따로 실행 (C228T, C250T, Wild).
run.sh 에서 3번 순차 호출됨.
"""

import os
import argparse
import json
import torch
import torch.distributed as dist
import numpy as np
import timm
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision import transforms


def build_model(device):
    model = timm.create_model(
        "hf-hub:bioptimus/H-optimus-0",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=False,
    )
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.707223, 0.578729, 0.703617),
            std=(0.211883, 0.230117, 0.177517),
        ),
    ])
    return model, transform


def is_valid_npy(path: Path) -> bool:
    """1KB 이상이면 완료된 파일로 판단."""
    return path.exists() and path.stat().st_size >= 1024


def embed_batch(img_paths, model, transform, device):
    imgs = []
    valid_paths = []
    for p in img_paths:
        try:
            img = Image.open(p).convert("RGB")
            imgs.append(transform(img))
            valid_paths.append(p)
        except Exception as e:
            print(f"[WARN] 타일 로드 실패, 건너뜀: {p} ({e})")
    if not imgs:
        return np.empty((0, 1536)), valid_paths
    tensor = torch.stack(imgs).to(device)
    with torch.no_grad():
        feats = model(tensor)
    return feats.cpu().numpy(), valid_paths


def process_slide(slide_dir, args, model, transform, device, rank, world_size):
    tile_paths = sorted(slide_dir.glob("*.png"))
    if len(tile_paths) == 0:
        if rank == 0:
            print(f"[WARN] 타일 없음, 건너뜀: {slide_dir}")
        return None

    sub_paths = tile_paths[rank::world_size]
    features, tiles_info = [], []

    for i in tqdm(
        range(0, len(sub_paths), args.batch_size),
        desc=f"GPU {rank} - {slide_dir.name}",
        disable=(rank != 0),
    ):
        batch_paths = sub_paths[i : i + args.batch_size]
        batch_info = []
        for p in batch_paths:
            parts = p.stem.split("_")
            try:
                x = int(parts[-2][1:])
                y = int(parts[-1][1:])
            except Exception:
                x, y = 0, 0
            batch_info.append({"name": p.name, "x": x, "y": y})

        feats, valid_paths = embed_batch(batch_paths, model, transform, device)
        # valid_paths 기준으로 info 재조정
        valid_names = {p.name for p in valid_paths}
        batch_info = [info for info in batch_info if info["name"] in valid_names]

        if feats.shape[0] > 0:
            features.append(feats)
        tiles_info.extend(batch_info)

    embed_dim = features[0].shape[-1] if features else 1536
    features = np.vstack(features) if features else np.empty((0, embed_dim))

    all_features = [None] * world_size
    all_info = [None] * world_size

    dist.all_gather_object(all_features, features)
    dist.all_gather_object(all_info, tiles_info)

    if rank == 0:
        merged_feats = [f for f in all_features if f is not None and f.shape[0] > 0]
        if not merged_feats:
            print(f"[WARN] {slide_dir.name}: 유효한 피처 없음, 건너뜀")
            return None
        all_features = np.vstack(merged_feats)
        all_info = sum([i for i in all_info if i is not None], [])

        slide_id = slide_dir.name

        npy_dir = Path(args.out_dir) / "npy"
        npy_dir.mkdir(parents=True, exist_ok=True)
        save_npy = npy_dir / f"{slide_id}.npy"
        np.save(save_npy, all_features)
        print(f"[✓] {save_npy}  shape={all_features.shape}")

        json_dir = Path(args.out_dir) / "json"
        json_dir.mkdir(parents=True, exist_ok=True)
        save_json = json_dir / f"{slide_id}.json"
        coord_data = {
            "slide_id": slide_id,
            "num_tiles": len(tile_paths),
            "embedding_path": str(save_npy),
            "patch_coords": all_info,
        }
        with open(save_json, "w") as f:
            json.dump(coord_data, f, indent=2)

        return coord_data
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile_dir", type=str, required=True,
                        help="클래스 타일 디렉토리 (예: .../40x_patch/C228T)")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="클래스 출력 디렉토리 (예: .../h_optimus_embeddings/C228T)")
    parser.add_argument("--batch_size", type=int, default=448)
    parser.add_argument("--slide_list", type=str, default=None,
                        help="처리할 슬라이드 ID 목록 txt (없으면 전체 처리)")
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    model, transform = build_model(device)

    tile_dir = Path(args.tile_dir)

    # resume 필터링: rank 0에서만 수행 후 broadcast
    if rank == 0:
        slide_dirs = sorted([p for p in tile_dir.iterdir() if p.is_dir()])
        if args.slide_list:
            with open(args.slide_list) as f:
                allowed = set(l.strip() for l in f if l.strip())
            slide_dirs = [p for p in slide_dirs if p.name in allowed]
            print(f"[INFO] slide_list 필터링: {len(allowed)}개 지정 → {len(slide_dirs)}개 매칭")

        npy_dir = Path(args.out_dir) / "npy"
        todo, skip = [], []
        for sd in slide_dirs:
            out_npy = npy_dir / f"{sd.name}.npy"
            if is_valid_npy(out_npy):
                skip.append(sd.name)
            else:
                todo.append(sd)

        if skip:
            print(f"[SKIP] 이미 완료된 슬라이드 {len(skip)}개 건너뜀")
        print(f"[INFO] 처리 대상: {len(todo)}/{len(todo)+len(skip)} 슬라이드")
        slide_dirs = todo
    else:
        slide_dirs = None

    slide_dirs_list = [slide_dirs]
    dist.broadcast_object_list(slide_dirs_list, src=0)
    slide_dirs = slide_dirs_list[0]

    master_data = []
    try:
        for idx, slide_dir in enumerate(slide_dirs):
            if rank == 0:
                print(f"\n[{idx+1}/{len(slide_dirs)}] {slide_dir.name}")

            coord_data = process_slide(
                slide_dir, args, model, transform, device, rank, world_size
            )
            if coord_data:
                master_data.append(coord_data)
    finally:
        dist.destroy_process_group()

    if rank == 0 and master_data:
        json_dir = Path(args.out_dir) / "json"
        all_json_path = json_dir / "all_slides.json"

        # 기존 all_slides.json 이 있으면 merge (resume 시 누적)
        existing = []
        if all_json_path.exists():
            with open(all_json_path) as f:
                existing = json.load(f)
            existing_ids = {d["slide_id"] for d in existing}
            master_data = existing + [d for d in master_data if d["slide_id"] not in existing_ids]

        with open(all_json_path, "w") as f:
            json.dump(master_data, f, indent=2)

        slide_ids = [item["slide_id"] for item in master_data]
        with open(json_dir / "slide_ids.txt", "w") as f:
            f.write("\n".join(slide_ids))

        print(f"\n[완료] 총 {len(slide_ids)}개 슬라이드")
        print(f"[INFO] npy: {args.out_dir}/npy/")
        print(f"[INFO] json: {args.out_dir}/json/")


if __name__ == "__main__":
    main()
