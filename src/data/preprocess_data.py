"""
Thyroid TERT 데이터 임베딩 (UNI2-h) - DataParallel 버전

# C228T
python preprocess_data.py \
    --tile_dir /path/to/40x_patch/C228T \
    --out_dir /path/to/embedding/C228T \
    --batch_size 1024

# C250T
python preprocess_data.py \
    --tile_dir /path/to/40x_patch/C250T \
    --out_dir /path/to/embedding/C250T \
    --batch_size 1024

# WT
python preprocess_data.py \
    --tile_dir /path/to/40x_patch/Wild \
    --out_dir /path/to/embedding/Wild \
    --batch_size 1024
"""

import os
import argparse
import torch
import torch.nn as nn
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
import timm


def build_model(device_ids):
    timm_kwargs = {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }
    model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    model.eval()

    # DataParallel로 멀티 GPU 사용
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    model.to(f"cuda:{device_ids[0]}")

    # transform 설정
    base_model = model.module if hasattr(model, 'module') else model
    data_cfg = resolve_data_config({}, model=base_model)
    transform = create_transform(**data_cfg)

    return model, transform


def embed_batch(img_paths, model, transform, device):
    imgs = []
    for p in img_paths:
        img = Image.open(p).convert("RGB")
        imgs.append(transform(img))
    tensor = torch.stack(imgs).to(device)
    with torch.no_grad():
        feats = model(tensor)
    return feats.cpu().numpy()


def process_slide(slide_dir, args, model, transform, device):
    slide_id = slide_dir.name
    npy_dir = Path(args.out_dir) / "npy"
    save_vec = npy_dir / f"{slide_id}.npy"

    # 이미 처리된 슬라이드는 건너뛰기
    if save_vec.exists():
        file_size = save_vec.stat().st_size
        # 1KB 미만이면 불완전한 파일로 판단하고 삭제
        if file_size < 1024:
            print(f"[DEL] {slide_id} - 불완전한 파일 삭제 ({file_size} bytes)")
            save_vec.unlink()
        else:
            print(f"[SKIP] {slide_id} - 이미 존재함")
            return None

    tile_paths = sorted(slide_dir.glob("*.png"))
    if len(tile_paths) == 0:
        return None

    features, tiles_info = [], []

    for i in tqdm(range(0, len(tile_paths), args.batch_size), desc=f"{slide_dir.name}"):
        batch_paths = tile_paths[i:i+args.batch_size]
        batch_info = []

        for p in batch_paths:
            parts = p.stem.split("_")
            try:
                x = int(parts[-2][1:])
                y = int(parts[-1][1:])
            except:
                x, y = 0, 0
            batch_info.append({"name": p.name, "x": x, "y": y})

        feats = embed_batch(batch_paths, model, transform, device)
        features.append(feats)
        tiles_info.extend(batch_info)

    features = np.vstack(features) if len(features) > 0 else np.empty((0, 1536))

    # npy 폴더에 임베딩 저장
    npy_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_vec, features)
    print(f"[✓] {save_vec} {features.shape}")

    # json 폴더에 패치 좌표 정보 저장
    json_dir = Path(args.out_dir) / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    save_json = json_dir / f"{slide_id}.json"

    coord_data = {
        "slide_id": slide_id,
        "num_tiles": len(tile_paths),
        "embedding_path": str(save_vec),
        "patch_coords": tiles_info
    }
    with open(save_json, "w") as f:
        json.dump(coord_data, f, indent=2)

    return coord_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gpus", type=str, default=None, help="GPU ids (e.g., '3,4,5')")
    args = parser.parse_args()

    # GPU 설정
    if args.gpus:
        device_ids = [int(x) for x in args.gpus.split(",")]
    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        device_ids = list(range(len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))))
    else:
        device_ids = [0]

    device = torch.device(f"cuda:{device_ids[0]}")
    print(f"[INFO] Using GPUs: {device_ids}")

    model, transform = build_model(device_ids)

    # 출력 디렉토리 생성
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    tile_dir = Path(args.tile_dir)
    slide_dirs = sorted([p for p in tile_dir.iterdir() if p.is_dir()])
    print(f"[INFO] 총 슬라이드 개수: {len(slide_dirs)}")

    master_data = []
    for idx, slide_dir in enumerate(slide_dirs):
        print(f"\n[{idx+1}/{len(slide_dirs)}] {slide_dir.name}")
        coord_data = process_slide(slide_dir, args, model, transform, device)
        if coord_data:
            master_data.append(coord_data)

    # 최종 저장
    if master_data:
        json_dir = Path(args.out_dir) / "json"

        # 전체 슬라이드 정보를 하나의 마스터 JSON에 저장
        all_slides_path = json_dir / "all_slides.json"
        with open(all_slides_path, "w") as f:
            json.dump(master_data, f, indent=2)

        slide_ids = [item["slide_id"] for item in master_data]
        slide_ids_path = json_dir / "slide_ids.txt"
        with open(slide_ids_path, "w") as f:
            f.write("\n".join(slide_ids))

        print(f"\n[✓✓✓] 완료! 총 {len(slide_ids)}개 슬라이드")
        print(f"[INFO] npy 저장: {args.out_dir}/npy/")
        print(f"[INFO] json 저장: {args.out_dir}/json/")


if __name__ == "__main__":
    main()
