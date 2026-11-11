#!/usr/bin/env python
"""
Virchow2 추출을 위한 환경 점검 — 추출 서버에서 먼저 실행할 것.

    python check_env.py                    # 점검만
    python check_env.py --download         # 가중치 다운로드까지 시도 (약 2.6GB)

점검 항목:
  1) torch / CUDA / GPU 목록과 compute capability (V100=sm_70, 3080=sm_86)
  2) timm 버전과 Virchow2 필수 기능 (SwiGLUPacked, reg_tokens, dynamic_img_size)
  3) openslide (C 라이브러리까지) — --source svs 에 필요
  4) HF 토큰 + Virchow2 gated 접근 권한
  5) 데이터/출력 경로 존재와 쓰기 권한, 남은 용량
  6) 아키텍처 실제 생성 + forward (가중치 없이) → 출력 차원 검증
  7) fp16 추론 속도 측정 (GPU별) → GPU_WEIGHTS 산출 근거
"""
import os as _os
# torch 의 GPU 번호를 nvidia-smi 와 일치시킨다(PCI 버스 순서).
# torch import 전에 설정해야 효과가 있다.
_os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
import argparse
import os
import shutil
import sys
import warnings

warnings.filterwarnings("ignore")

OK, BAD, WARN, INFO = "[O]", "[X]", "[!]", "[i]"
fails, warns = [], []


def ok(m):   print(f"  {OK} {m}")
def bad(m):  print(f"  {BAD} {m}"); fails.append(m)
def warn(m): print(f"  {WARN} {m}"); warns.append(m)
def info(m): print(f"  {INFO} {m}")
def head(t): print(f"\n=== {t} ===")


# ---------- 기본 경로 (TERT 전용. --out_root 등으로 덮어쓸 수 있음) ----------
# BRAF 용 체크는 Thyroid_Mutation_Prediction_Model/src/data/virchow2/check_env.py 를 쓴다.
TERT_ROOT = "/path/to/Thyroid_TERT_dataset"
DEF_OUT = "/path/to/embeddings/tert/virchow2_40x512_concat"
DEF_PATCH = f"{TERT_ROOT}/40x_patch"
DEF_SVS = [f"{TERT_ROOT}/thyroid"]
# TERT 는 클래스별 디렉터리 구조다 (WSI 수)
TERT_CLASSES = {"Wild": 113, "C228T": 69, "C250T": 19}


def check_torch():
    head("1) torch / GPU")
    try:
        import torch
    except Exception as e:
        bad(f"torch import 실패: {e}")
        return None
    ok(f"torch {torch.__version__}")
    if not torch.cuda.is_available():
        bad("CUDA 사용 불가 — 드라이버/torch 빌드 불일치 가능")
        return torch
    n = torch.cuda.device_count()
    ok(f"CUDA {torch.version.cuda} / GPU {n}장 (CUDA_VISIBLE_DEVICES="
       f"{os.environ.get('CUDA_VISIBLE_DEVICES', '미설정')})")
    bf16_all = True
    for i in range(n):
        cap = torch.cuda.get_device_capability(i)
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        sm = f"sm_{cap[0]}{cap[1]}"
        bf = cap[0] >= 8                      # Ampere 이상만 bf16 하드웨어 지원
        bf16_all &= bf
        print(f"       GPU{i}  {name:<34} {sm}  {mem:5.1f}GB  bf16={'O' if bf else 'X'}")
        if mem < 8:
            warn(f"GPU{i} VRAM {mem:.1f}GB — bs=256은 약 3.9GB 필요, 여유 확인")
    if not bf16_all:
        info("bf16 미지원 GPU 포함 → --dtype fp16 사용 (Virchow2 공식 설정도 fp16)")
    return torch


def check_timm():
    head("2) timm (Virchow2 필수 기능)")
    try:
        import timm
    except Exception as e:
        bad(f"timm import 실패: {e}")
        return False
    ok(f"timm {timm.__version__} (검증 버전: 1.0.29)")
    good = True
    try:
        from timm.layers import SwiGLUPacked           # noqa: F401
        ok("SwiGLUPacked")
    except Exception as e:
        bad(f"SwiGLUPacked 없음 ({e}) → timm 업그레이드 필요: pip install -U timm")
        good = False
    try:
        import inspect
        from timm.models.vision_transformer import VisionTransformer
        p = inspect.signature(VisionTransformer.__init__).parameters
        for need in ("reg_tokens", "dynamic_img_size"):
            if need in p:
                ok(f"{need} 인자 지원")
            else:
                bad(f"{need} 미지원 → timm 업그레이드 필요")
                good = False
    except Exception as e:
        bad(f"VisionTransformer 점검 실패: {e}")
        good = False
    return good


def check_openslide():
    head("3) openslide (--source svs 용)")
    try:
        import openslide
        ok(f"openslide {openslide.__version__} (C 라이브러리 포함)")
        return True
    except Exception as e:
        warn(f"openslide 사용 불가: {type(e).__name__}")
        info("--source png 로 돌리면 됩니다 (결과 동일, 약간 느림).")
        info("또는: pip install openslide-python openslide-bin")
        return False


def check_hf(download=False):
    head("4) HF 토큰 / Virchow2 접근 권한")
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except Exception as e:
        bad(f"huggingface_hub 없음: {e}")
        return False
    info(f"HF_HOME={os.environ.get('HF_HOME', '기본값(~/.cache/huggingface)')}")
    try:
        who = HfApi().whoami()
        ok(f"토큰 인증됨 (user: {who.get('name')})")
    except Exception:
        bad("HF 토큰 없음 → export HF_TOKEN=hf_xxx 또는 huggingface-cli login")
        return False
    try:
        hf_hub_download("paige-ai/Virchow2", "config.json")
        ok("paige-ai/Virchow2 접근 가능 (gated 승인 확인)")
    except Exception as e:
        bad(f"Virchow2 접근 불가 ({type(e).__name__}) — "
            "https://huggingface.co/paige-ai/Virchow2 에서 라이선스 동의 필요")
        return False
    if download:
        try:
            print("       가중치 다운로드 중 (약 2.6GB)...")
            p = hf_hub_download("paige-ai/Virchow2", "pytorch_model.bin")
            ok(f"가중치 확보: {os.path.getsize(p)/1e9:.2f}GB")
        except Exception as e:
            warn(f"가중치 다운로드 실패: {type(e).__name__} (추출 시 자동 재시도됨)")
    return True


def check_paths(out_root, patch_root, svs_dirs):
    head("5) 경로 / 용량")
    for d in svs_dirs:
        if os.path.isdir(d):
            ok(d)
        else:
            bad(f"경로 없음: {d}")
    # 클래스별 패치 디렉터리와 WSI(슬라이드 하위 디렉터리) 수를 확인한다
    total = 0
    for cls, expect in TERT_CLASSES.items():
        d = os.path.join(patch_root, cls)
        if not os.path.isdir(d):
            bad(f"클래스 패치 경로 없음: {d}")
            continue
        try:
            n = sum(1 for e in os.scandir(d) if e.is_dir())
        except Exception as e:
            bad(f"{d} 조회 실패: {e}")
            continue
        total += n
        if n == expect:
            ok(f"{cls}: {n} WSI")
        else:
            warn(f"{cls}: {n} WSI (기대 {expect})")
    if total == 201:
        ok(f"합계 {total} / 201 WSI")
    else:
        warn(f"합계 {total} / 201 WSI — 패치 추출이 덜 됐을 수 있다")
    try:
        os.makedirs(out_root, exist_ok=True)
        if os.access(out_root, os.W_OK):
            ok(f"쓰기 가능: {out_root}")
        else:
            bad(f"쓰기 불가: {out_root}")
    except Exception as e:
        bad(f"출력 디렉터리 생성 실패: {out_root} ({e})")
    try:
        free = shutil.disk_usage(out_root).free / 1e12
        # TERT 201장은 BRAF(4900장)의 약 1/24 → concat 기준 약 50GB 면 충분하다
        info(f"남은 용량 {free:.2f}TB (TERT 201장 concat 약 0.05TB 필요)")
        if free < 0.1:
            warn("용량이 빠듯하다 → EMBED_MODE=cls 고려")
    except Exception:
        pass


def check_arch(torch, speed=True):
    head("6) Virchow2 아키텍처 생성 + forward (가중치 없이)")
    if torch is None:
        bad("torch 없음 → 생략")
        return
    try:
        import timm
        from timm.layers import SwiGLUPacked
        kw = dict(img_size=224, init_values=1e-5, num_classes=0, reg_tokens=4,
                  mlp_ratio=5.3375, global_pool="", dynamic_img_size=True,
                  mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
        m = timm.create_model("vit_huge_patch14_224", pretrained=False, **kw).eval()
        n = sum(p.numel() for p in m.parameters()) / 1e6
        if abs(n - 631) < 5:
            ok(f"파라미터 {n:.0f}M (공식 631M 일치)")
        else:
            warn(f"파라미터 {n:.0f}M — 공식 631M과 다름, timm 버전 확인")
        if not torch.cuda.is_available():
            warn("CUDA 불가 → forward/속도 측정 생략")
            return
        m = m.to("cuda:0", dtype=torch.float16)
        with torch.inference_mode():
            out = m(torch.randn(2, 3, 224, 224, device="cuda:0", dtype=torch.float16))
        if tuple(out.shape) == (2, 261, 1280):
            ok(f"출력 {tuple(out.shape)} = 1 CLS + 4 register + 256 patch")
        else:
            bad(f"출력 {tuple(out.shape)} — 기대 (B, 261, 1280)")
        cls = out[:, 0]
        cat = torch.cat([cls, out[:, 5:].mean(1)], -1)
        ok(f"concat {tuple(cat.shape)} (2560) / cls {tuple(cls.shape)} (1280)")
        import numpy as np
        f = cat.float().cpu().numpy()
        if np.isnan(f).any() or np.isinf(f).any():
            bad("출력에 NaN/Inf — fp16 불안정 가능")
        else:
            ok("NaN/Inf 없음")

        if speed:
            head("7) GPU별 fp16 처리량 (GPU_WEIGHTS 산출 근거) — 번호는 nvidia-smi 기준")
            import time
            res = []
            # 주의: m 은 float32 로 생성돼 있다. .to(device) 는 dtype 을 바꾸지
            # 않으므로 fp16 입력을 넣으면 dtype mismatch 로 터진다.
            # 반드시 .to(device=..., dtype=float16) 로 모델까지 반정밀도로 옮긴다.
            for i in range(torch.cuda.device_count()):
                dev = f"cuda:{i}"
                try:
                    # V100S(sm_70)는 FlashAttention 미지원 → 백엔드 전환 필요.
                    # (실제 추출 코드도 build_model 에서 동일 처리를 한다)
                    cap = torch.cuda.get_device_capability(i)[0]
                    torch.backends.cuda.enable_flash_sdp(cap >= 8)
                    torch.backends.cuda.enable_mem_efficient_sdp(True)
                    torch.backends.cuda.enable_math_sdp(True)
                    mm = m.to(device=dev, dtype=torch.float16).eval()
                    x = torch.randn(32, 3, 224, 224, device=dev, dtype=torch.float16)
                    with torch.inference_mode():
                        for _ in range(2):
                            mm(x)
                        torch.cuda.synchronize(i)
                        t = time.time()
                        for _ in range(3):
                            mm(x)
                        torch.cuda.synchronize(i)
                        dt = (time.time() - t) / 3
                    ips = 32 / dt
                    res.append(ips)
                    print(f"       GPU{i}  {ips:6.1f} patch/s  "
                          f"({torch.cuda.get_device_name(i)})")
                except Exception as e:
                    res.append(0.0)
                    warn(f"GPU{i} 측정 실패: {type(e).__name__}: {e}")
                finally:
                    torch.cuda.empty_cache()
            if res and max(res) > 0:
                mx = max(res)
                w = ",".join(f"{r/mx:.2f}" for r in res)
                print()
                ok(f"권장 GPU_WEIGHTS=\"{w}\"")
                info("run_mixed_gpu.sh 의 GPU_WEIGHTS 를 위 값으로 바꾸면 분배가 최적화됩니다.")
                tot = sum(res)
                if tot > 0:
                    info(f"합계 {tot:.0f} patch/s → BRAF 4900장(평균 18000 tile) "
                         f"약 {18000*4900/tot/3600:.0f}시간 예상")
    except Exception as e:
        import traceback
        bad(f"아키텍처 점검 실패: {type(e).__name__}: {e}")
        traceback.print_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=DEF_OUT)
    ap.add_argument("--patch_root", default=DEF_PATCH)
    ap.add_argument("--svs_dirs", nargs="*", default=DEF_SVS)
    ap.add_argument("--download", action="store_true", help="가중치까지 받아본다")
    ap.add_argument("--no_speed", action="store_true", help="속도 측정 생략")
    a = ap.parse_args()

    print("=" * 62)
    print(" Virchow2 환경 점검")
    print(f" python: {sys.executable}")
    print("=" * 62)

    torch = check_torch()
    check_timm()
    check_openslide()
    check_hf(a.download)
    check_paths(a.out_root, a.patch_root, a.svs_dirs)
    check_arch(torch, speed=not a.no_speed)

    print("\n" + "=" * 62)
    if fails:
        print(f" 결과: 실패 {len(fails)}건 — 해결 후 추출을 시작하세요")
        for f in fails:
            print(f"   {BAD} {f}")
    else:
        print(" 결과: 필수 항목 통과 — 추출 가능")
    if warns:
        print(f" 경고 {len(warns)}건:")
        for w in warns:
            print(f"   {WARN} {w}")
    print("=" * 62)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
