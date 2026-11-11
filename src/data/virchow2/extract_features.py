"""
Virchow2 Feature Extraction (DDP)

Paige AI Virchow2 (ViT-H/14, 631M). PathBench 2위 / Nature BME 2026 biomarker 공동 1위.
GenBio 추출기와 DDP/저장 로직은 동일하고 인코더만 교체했다.

임베딩 차원 — 두 가지를 고를 수 있다 (--embed_mode):
  concat (기본, 2560) : concat(CLS 1280, mean(patch) 1280). Paige 공식 권장 방식.
  cls    (1280)       : CLS 토큰만.
  Nature BME 2026은 다른 FM과의 일관성을 위해 cls(1280)로 평가했고,
  "두 설정의 차이는 매우 작다(very small)"고 보고했다.
  → 저장공간이 부담되면 cls(1280)도 합리적 선택이다 (용량 절반).

주의사항:
  1) 학습 시 --in_dim 을 반드시 맞출 것 (concat=2560 / cls=1280).
  2) register token 4개를 반드시 건너뛴다: output[:, 5:] (0=CLS, 1~4=register, 5~=patch 256개).
     출력은 [B, 261, 1280] = 1 CLS + 4 reg + 256 patch. (실측 검증 2026-11-05)
  3) timm_kwargs 가 특수하다 — reg_tokens=4, mlp_ratio=5.3375, SwiGLUPacked, SiLU.
     빠뜨리면 가중치 로드가 실패하거나 조용히 잘못된 출력이 나온다.
  4) Normalize 는 ImageNet 값이다 (0.485,0.456,0.406)/(0.229,0.224,0.225).
     H-optimus/Hibou/GenBio 값과 다르다. 복사하지 말 것.
     (출처: TRIDENT 0.3.2 Virchow2InferenceEncoder 구현)
  5) gated repo — https://huggingface.co/paige-ai/Virchow2 에서 라이선스 동의 후
     `huggingface-cli login` 또는 HF_TOKEN 필요. (승인은 auto, 클릭 즉시)

라이선스: CC-BY-NC-ND 4.0 — 비상업 연구용. 상업 사용은 Paige와 별도 협의 필요.

torchrun --nproc_per_node=8 --master_port=29505 extract_features.py \\
    --tile_dir /path/to/patches --out_dir /path/to/out \\
    --source svs --svs_dirs /path/to/svs --embed_mode concat
"""

import datetime
import gc
import os
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

# DataLoader worker가 공유 텐서를 파일 디스크립터로 주고받는 기본 방식은
# 타일 수가 많은 슬라이드에서 fd 한도를 넘겨 "Too many open files"로 죽는다.
# 파일시스템 기반 공유로 바꾸면 fd를 소모하지 않는다.
torch.multiprocessing.set_sharing_strategy("file_system")

# fd 한도를 하드 리밋까지 올린다. rank 5개 x worker 16개가 슬라이드마다
# Queue/Lock 을 만들기 때문에 기본값(보통 1024)으로는 금방 바닥난다.
try:
    import resource as _resource
    _soft, _hard = _resource.getrlimit(_resource.RLIMIT_NOFILE)
    if _soft < _hard:
        _resource.setrlimit(_resource.RLIMIT_NOFILE, (_hard, _hard))
except Exception:
    pass

# Ada 세대에서 matmul을 TF32로 허용 (정확도 영향 미미, 속도 이득)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

MODEL_DTYPE = torch.float16    # --dtype 로 변경 (Virchow2는 fp16이 공식 설정)
EMBED_MODE = "concat"          # concat(2560) | cls(1280)
from pathlib import Path
from PIL import Image
import re
import numpy as np
from tqdm import tqdm
import json
from torchvision import transforms


def disable_flash_on_volta(device):
    """sm_80 미만(V100/V100S 등 Volta)에서 FlashAttention SDPA 백엔드를 끈다.

    PyTorch SDPA 는 기본적으로 FlashAttention 을 고르는데 Flash 는 Ampere(sm_80)
    이상에서만 동작한다. V100S 에서 그대로 두면 forward 중에
        RuntimeError: FlashAttention only supports Ampere GPUs or newer.
    로 죽는다. Flash 를 끄고 mem_efficient / math 로 넘기면 정상 동작하며,
    애초에 Flash 를 쓸 수 없는 하드웨어라 속도 손해도 없다.
    """
    if not torch.cuda.is_available():
        return
    try:
        major, minor = torch.cuda.get_device_capability(device)
    except Exception:
        return
    if major >= 8:
        return
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        print(f"[i] {torch.cuda.get_device_name(device)} (sm_{major}{minor}) "
              f"→ FlashAttention 비활성화, mem_efficient/math SDPA 사용", flush=True)
    except Exception as e:
        print(f"[!] SDPA 백엔드 설정 실패(무시하고 진행): {e}", flush=True)


def build_model(device, model_name="paige-ai/Virchow2", dtype=torch.float16):
    """Virchow2 로드. timm_kwargs는 Paige 공식 / TRIDENT 구현과 동일하게 맞췄다."""
    import timm
    from timm.layers import SwiGLUPacked

    timm_kwargs = {
        "img_size": 224,
        "init_values": 1e-5,
        "num_classes": 0,
        "reg_tokens": 4,          # register token 4개
        "mlp_ratio": 5.3375,      # Virchow2 전용 값
        "global_pool": "",        # 토큰을 직접 다루므로 pooling 끔
        "dynamic_img_size": True,
        "mlp_layer": SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
    }
    model = timm.create_model(f"hf-hub:{model_name}", pretrained=True, **timm_kwargs)
    model.to(device, dtype=dtype)
    model.eval()
    disable_flash_on_volta(device)   # V100S(sm_70) 대응

    # ImageNet 정규화 — Virchow2는 이 값을 쓴다 (다른 FM과 다름).
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])
    return model, transform


class TileDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        p = self.img_paths[idx]
        img = Image.open(p).convert("RGB")
        tensor = self.transform(img)

        parts = p.stem.split("_")
        try:
            x = int(parts[-2][1:])
            y = int(parts[-1][1:])
        except:
            x, y = 0, 0
        return tensor, x, y, p.name


class SvsRegionDataset(Dataset):
    """WSI(.svs)에서 좌표로 직접 region을 읽는다. 중간 PNG가 필요 없다.

    실측(RTX 6000 Ada 서버, 2026-11-05): num_workers=16에서 약 1020 patch/s로
    PNG 디코드(843 patch/s)보다 빠르다. 단 worker가 적으면 역전된다
    (workers=8이면 SVS 162 vs PNG 452) — read_region이 worker 병렬화에
    크게 의존하므로 --num_workers를 16 이상으로 줄 것.

    OpenSlide 핸들은 fork-safe하지 않아 worker 안에서 lazy open한다.
    """

    def __init__(self, svs_path, coords, patch_size_src, transform):
        self.svs_path = str(svs_path)
        self.coords = coords            # [(x, y, name), ...]
        self.patch_size_src = patch_size_src
        self.transform = transform
        self._slide = None

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        if self._slide is None:
            import openslide
            self._slide = openslide.OpenSlide(self.svs_path)
        x, y, name = self.coords[idx]
        img = self._slide.read_region(
            (x, y), 0, (self.patch_size_src, self.patch_size_src)
        ).convert("RGB")
        return self.transform(img), x, y, name


def shard_weighted(items, rank, world_size, weights=None):
    """작업을 rank별로 나눈다. weights를 주면 성능 비례로 불균등 분배한다.

    혼종 GPU 서버(예: V100S 32GB + RTX 3080 10GB)에서는 균등 분배(items[rank::ws])가
    비효율적이다. all_gather가 전원 동기이므로 가장 느린 GPU가 전체 속도를 결정한다.
    --gpu_weights "1,1,1,0.7,0.7" 처럼 주면 느린 GPU에 적은 몫을 준다.

    weights=None 이면 기존 동작(균등 stride)과 동일하다.
    """
    n = len(items)
    if not weights or len(weights) != world_size:
        return items[rank::world_size]

    total = float(sum(weights))
    if total <= 0:
        return items[rank::world_size]

    # 누적 비율로 연속 구간을 할당한다 (SVS read_region은 지역성이 있어 연속이 유리).
    start = int(round(n * sum(weights[:rank]) / total))
    end = int(round(n * sum(weights[:rank + 1]) / total))
    if rank == world_size - 1:
        end = n
    return items[start:end]


def coords_from_patch_dir(slide_dir):
    """패치 PNG 파일명(..._x12345_y67890.png)에서 좌표만 수집한다.

    이미지를 열지 않고 파일명만 읽으므로 빠르다. PNG가 좌표 목록 역할만 한다.
    """
    coords = []
    for entry in os.scandir(slide_dir):
        if not entry.name.endswith(".png"):
            continue
        m = re.search(r"_x(\d+)_y(\d+)", entry.name)
        if m:
            coords.append((int(m.group(1)), int(m.group(2)), entry.name))
    coords.sort(key=lambda c: (c[1], c[0]))
    return coords


def find_svs(svs_dirs, slide_id):
    """slide_id에 해당하는 .svs를 주어진 디렉터리들에서 찾는다."""
    for d in svs_dirs:
        cand = Path(d) / f"{slide_id}.svs"
        if cand.exists():
            return cand
    return None


def _run_loader(dataset, n_tiles, slide_dir, args, model, device, rank, world_size):
    """dataset을 돌려 임베딩을 추출하고 rank0에서 npy/json으로 저장한다.

    PNG 경로와 SVS 경로가 이 함수를 공유한다.
    """
    # persistent_workers 는 반드시 False 로 둔다.
    # 이 함수는 슬라이드마다 새 DataLoader 를 만드는데, persistent_workers=True 면
    # worker 프로세스와 그에 딸린 Queue/Lock 의 fd 가 슬라이드마다 쌓여
    #   OSError: [Errno 24] Too many open files
    # 로 전 rank 가 죽는다 (실제로 27장째에서 발생).
    # 슬라이드당 1회만 순회하므로 persistent 로 얻을 이득도 없다.
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False,
    )

    features, coords, tiles_info = [], [], []

    for batch_tensor, xs, ys, names in tqdm(
            loader,
            desc=f"GPU {rank} - {slide_dir.name}" if rank == 0 else None,
            disable=(rank != 0)):
        batch_tensor = batch_tensor.to(device, dtype=MODEL_DTYPE, non_blocking=True)
        with torch.no_grad():
            out = model(batch_tensor)          # [B, 261, 1280]
        # 토큰 배치: 0=CLS, 1~4=register(반드시 제외), 5~=patch(256개, patch14 -> (224/14)^2)
        class_token = out[:, 0]                                  # [B, 1280]
        if EMBED_MODE == "cls":
            feats = class_token                                  # 1280
        else:
            patch_tokens = out[:, 5:]                            # [B, 256, 1280]
            feats = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)   # 2560
        features.append(feats.float().cpu().numpy())

        for x, y, name in zip(xs.tolist(), ys.tolist(), names):
            coords.append([x, y])
            tiles_info.append({"name": name, "x": x, "y": y})

    # 빈 슬라이드일 때의 fallback. 차원을 하드코딩하지 않는다 (concat=2560 / cls=1280).
    embed_dim = features[0].shape[-1] if features else 0
    features = np.vstack(features) if features else np.empty((0, embed_dim))

    all_features = [None] * world_size
    all_coords = [None] * world_size
    all_info = [None] * world_size

    # worker 와 fd 를 즉시 반납한다.
    # persistent_workers=False 이므로 순회가 끝나면 _MultiProcessingDataLoaderIter
    # 가 스스로 worker 를 정리하지만, loader 객체 자체를 빨리 버려야 다음 슬라이드
    # DataLoader 가 만들어지기 전에 fd 가 확실히 회수된다.
    # (loader._iterator 는 persistent_workers=True 일 때만 설정되므로 건드리지 않는다.)
    del loader, dataset
    gc.collect()

    dist.all_gather_object(all_features, features)
    dist.all_gather_object(all_coords, coords)
    dist.all_gather_object(all_info, tiles_info)

    if rank == 0:
        all_features = np.vstack([f for f in all_features if f is not None and f.shape[0] > 0])
        all_coords = sum([c for c in all_coords if c is not None], [])
        all_info = sum([i for i in all_info if i is not None], [])

        slide_id = slide_dir.name

        npy_dir = Path(args.out_dir) / "npy"
        npy_dir.mkdir(parents=True, exist_ok=True)
        save_vec = npy_dir / f"{slide_id}.npy"
        np.save(save_vec, all_features)
        print(f"[✓] {save_vec} {all_features.shape}")

        json_dir = Path(args.out_dir) / "json"
        json_dir.mkdir(parents=True, exist_ok=True)
        save_json = json_dir / f"{slide_id}.json"

        coord_data = {
            "slide_id": slide_id,
            "num_tiles": n_tiles,
            "embedding_path": str(save_vec),
            "patch_coords": all_info,
        }
        with open(save_json, "w") as f:
            json.dump(coord_data, f, indent=2)

        return coord_data


def process_slide(slide_dir, args, model, transform, device, rank, world_size):
    # --source svs : WSI에서 좌표로 직접 읽는다 (중간 PNG 디코드 생략)
    # --source png : 패치 PNG를 읽는다 (기본, 기존 동작)
    if args.source == "svs":
        coords = coords_from_patch_dir(slide_dir)
        svs_path = find_svs(args.svs_dirs, slide_dir.name) if coords else None
        if coords and svs_path is not None:
            sub = shard_weighted(coords, rank, world_size, args.gpu_weights)
            dataset = SvsRegionDataset(svs_path, sub, args.patch_size_src, transform)
            return _run_loader(dataset, len(coords), slide_dir, args, model,
                               device, rank, world_size)
        if rank == 0 and coords:
            print(f"[!] .svs 미발견 → PNG 경로로 대체: {slide_dir.name}")

    tile_paths = sorted(slide_dir.glob("*.png"))
    if len(tile_paths) == 0:
        return None
    sub_paths = shard_weighted(tile_paths, rank, world_size, args.gpu_weights)
    dataset = TileDataset(sub_paths, transform)
    return _run_loader(dataset, len(tile_paths), slide_dir, args, model,
                       device, rank, world_size)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="paige-ai/Virchow2",
                        help="paige-ai/Virchow2 (concat=2560 / cls=1280)")
    parser.add_argument("--embed_mode", type=str, default="concat", choices=["concat", "cls"],
                        help="concat=CLS+mean(patch) 2560 (공식 권장) | cls=CLS만 1280")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["bf16", "fp16"],
                        help="Virchow2 공식 설정은 fp16")
    parser.add_argument("--gpu_weights", type=str, default=None,
                        help="혼종 GPU 서버용 성능 가중치. 예: '1,1,1,0.7,0.7' "
                             "(V100S 3장 + RTX3080 2장). 생략하면 균등 분배")
    parser.add_argument("--source", type=str, default="png", choices=["png", "svs"],
                        help="png=패치 PNG 읽기(기본) | svs=WSI에서 좌표로 직접 읽기")
    parser.add_argument("--svs_dirs", type=str, nargs="*", default=[],
                        help="--source svs 일 때 .svs 를 찾을 디렉터리들")
    parser.add_argument("--patch_size_src", type=int, default=512,
                        help="--source svs 일 때 WSI에서 떼올 원본 패치 크기 (40x 512)")
    parser.add_argument("--batch_size", type=str, default="256",
                        help="배치 크기. 단일값('256') 또는 GPU별 쉼표 구분"
                             "('192,192,512,512,512' — GPUS 나열 순서와 1:1). "
                             "VRAM 이 다른 GPU를 섞어 쓸 때 큰 쪽에 더 준다.")
    parser.add_argument("--auto_batch", action="store_true",
                        help="각 GPU의 VRAM을 보고 batch_size를 자동 결정한다. "
                             "--batch_size 보다 우선한다.")
    parser.add_argument("--vram_frac", type=float, default=0.80,
                        help="--auto_batch 시 사용할 VRAM 비율 (기본 0.80)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader worker 프로세스 수 (이미지 로딩 병렬화)")
    parser.add_argument("--slide_list", type=str, default=None,
                        help="슬라이드 ID 목록 txt (한 줄에 하나, 없으면 전체 처리)")
    args = parser.parse_args()

    # NCCL 타임아웃을 기본 10분에서 2시간으로 늘린다.
    # 이 코드는 슬라이드마다 all_gather_object 로 전 rank 를 동기화하는데,
    # 한 rank 가 느린 슬라이드(큰 SVS, NFS 지연)를 만나면 나머지가 그만큼 대기한다.
    # 2026-11-09 04:52 실패: 배치 1개에 362초 걸린 슬라이드에서 rank4 가
    # 10분 한도를 넘겨 SIGABRT 로 전체가 내려갔다 (1890/4900 진행 상태).
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # gpu_weights: "1,1,1,0.7,0.7" -> [1.0,1.0,1.0,0.7,0.7]
    if isinstance(args.gpu_weights, str) and args.gpu_weights.strip():
        try:
            args.gpu_weights = [float(w) for w in args.gpu_weights.split(",")]
        except ValueError:
            if rank == 0:
                print(f"[!] --gpu_weights 파싱 실패 → 균등 분배: {args.gpu_weights}")
            args.gpu_weights = None
    else:
        args.gpu_weights = None
    if args.gpu_weights and len(args.gpu_weights) != world_size:
        if rank == 0:
            print(f"[!] --gpu_weights 개수({len(args.gpu_weights)}) != GPU 수({world_size}) "
                  f"→ 균등 분배")
        args.gpu_weights = None

    # ---- batch_size 를 rank 별로 결정 --------------------------------------
    # VRAM 이 다른 GPU(3080 10GB / V100S 32GB)를 섞어 쓸 때, 모든 GPU가 같은
    # 배치를 쓰면 작은 쪽에 맞춰야 해서 큰 쪽의 메모리가 놀게 된다.
    # 여기서 rank 마다 다른 배치를 정한다. 결과값은 배치 크기와 무관하게 동일하다.
    _bs_raw = str(args.batch_size).strip()
    if args.auto_batch:
        # 실측(2026-11-06, Virchow2 fp16, RTX6000 Ada):
        #   가중치 1.18GB 고정, activation 은 배치에 완전히 선형
        #   bs 128/256/512/1024 -> 1.22/2.42/4.83/9.65GB  = 타일당 9.64MB
        # 여기에 CUDA 컨텍스트와 DataLoader pinned buffer 몫으로 1.0GB 를 더 뺀다.
        total_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
        # 2026-11-06 실패 기록: reserved 2.18GB / 타일당 9.64MB 로 잡았다가
        # V100S bs=2464 에서 OOM. 실제로는 (a) 단편화로 '예약했지만 못 쓰는'
        # 메모리가 5.7GB 나 생기고 (b) 같은 GPU 를 쓰는 타 프로세스가 있었다.
        # 그래서 고정분을 크게, 타일당 비용도 단편화 계수 1.3배로 올려 잡는다.
        reserved_gb = 1.18 + 3.0               # 가중치 + 컨텍스트 + 타 프로세스 여유
        per_tile_gb = 9.64 / 1024 * 1.3        # 실측 기울기 x 단편화 여유
        usable = max(total_gb * args.vram_frac - reserved_gb, 0.5)
        bs = int(usable / per_tile_gb)
        bs = max(32, (bs // 32) * 32)          # 32 단위로 내림
        bs = min(bs, 4096)                     # 상한
        args.batch_size = bs
        print(f"[auto_batch] rank{rank} {torch.cuda.get_device_name(device)} "
              f"{total_gb:.1f}GB -> batch_size={bs}", flush=True)
    elif "," in _bs_raw:
        try:
            bs_list = [int(b) for b in _bs_raw.split(",")]
        except ValueError:
            bs_list = []
        if len(bs_list) == world_size:
            args.batch_size = bs_list[rank]
        else:
            if rank == 0:
                print(f"[!] --batch_size 개수({len(bs_list)}) != GPU 수({world_size}) "
                      f"-> 첫 값 사용", flush=True)
            args.batch_size = bs_list[0] if bs_list else 256
    else:
        args.batch_size = int(_bs_raw)
    # ------------------------------------------------------------------------

    global MODEL_DTYPE, EMBED_MODE
    MODEL_DTYPE = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    EMBED_MODE = args.embed_mode

    # bf16은 Ampere(sm_80) 이상에서만 하드웨어 지원된다. V100(Volta, sm_70)은 미지원이라
    # 매우 느려지거나 실패한다. 지원 여부를 확인해 fp16으로 자동 강등한다.
    if MODEL_DTYPE is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        if rank == 0:
            cap = torch.cuda.get_device_capability(device)
            print(f"[!] bf16 미지원 GPU (sm_{cap[0]}{cap[1]}) → fp16으로 전환")
        MODEL_DTYPE = torch.float16
    model, transform = build_model(device, args.model_name, MODEL_DTYPE)

    tile_dir = Path(args.tile_dir)
    if rank == 0:
        slide_dirs = sorted([p for p in tile_dir.iterdir() if p.is_dir()])
        if args.slide_list:
            with open(args.slide_list) as f:
                allowed = set(l.strip() for l in f if l.strip())
            slide_dirs = [p for p in slide_dirs if p.name in allowed]
            print(f"[INFO] slide_list 필터링: {len(allowed)}개 지정 → {len(slide_dirs)}개 매칭")
        # resume: 이미 완료된 슬라이드 skip
        npy_dir = Path(args.out_dir) / "npy"
        done = {p.stem for p in npy_dir.glob("*.npy")} if npy_dir.exists() else set()
        if done:
            before = len(slide_dirs)
            slide_dirs = [p for p in slide_dirs if p.name not in done]
            print(f"[INFO] resume: {before}개 중 {len(done)}개 완료 → 나머지 {len(slide_dirs)}개 처리")
        print(f"[INFO] 총 슬라이드 개수: {len(slide_dirs)}")
    else:
        slide_dirs = None

    slide_dirs_list = [slide_dirs]
    dist.broadcast_object_list(slide_dirs_list, src=0)
    slide_dirs = slide_dirs_list[0]

    master_data = []
    for idx, slide_dir in enumerate(slide_dirs):
        if rank == 0:
            print(f"\n[{idx+1}/{len(slide_dirs)}] {slide_dir.name}")

        coord_data = process_slide(slide_dir, args, model, transform, device, rank, world_size)
        if coord_data:
            master_data.append(coord_data)

    if rank == 0 and master_data:
        json_dir = Path(args.out_dir) / "json"
        with open(json_dir / "all_slides.json", "w") as f:
            json.dump(master_data, f, indent=2)

        slide_ids = [item["slide_id"] for item in master_data]
        with open(json_dir / "slide_ids.txt", "w") as f:
            f.write("\n".join(slide_ids))

        print(f"\n[✓✓✓] 완료! 총 {len(slide_ids)}개 슬라이드")
        print(f"[INFO] npy 저장: {args.out_dir}/npy/")
        print(f"[INFO] json 저장: {args.out_dir}/json/")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
