# Thyroid TERT Promoter Mutation Prediction (공개용)

WSI 패치 임베딩 기반으로 TERT promoter mutation을 예측하는 공개용 MIL 학습 파이프라인입니다.

![TERT Pipeline](images/tert_pipeline.png)

## 1. 문제 정의
- 태스크: 이진 분류 (`Wild=0`, `Mutant(C228T/C250T)=1`)
- 입력: 슬라이드 단위 패치 임베딩 배열(`.npy`)
- 출력: 5-fold CV 성능 지표, 체크포인트, 시각화, attention heatmap
- 지원 모델: `ABMIL`, `TransMIL`

## 2. 저장소 구조
- `src/data/preprocess_data.py`: 패치 이미지 -> 임베딩(`UNI2-h`)
- `src/data/run_embedding.sh`: 서브타입별 임베딩 병렬 실행 헬퍼
- `src/data/create_cv_splits.py`: Stratified K-fold split 생성
- `src/training/main.py`: 학습 엔트리포인트
- `src/training/train_tert.py`: CV 학습/평가 핵심 파이프라인
- `src/training/train.sh`: 환경변수 오버라이드 가능한 실행 스크립트
- `src/training/mlflow_utils.py`: 선택적 MLflow 로깅
- `src/training/register_model.py`: 선택적 모델 등록
- `src/evaluation/visualization.py`: attention heatmap 생성
- `config/cv_splits_tert_example.json`: CV split 예시 포맷
- `config/tert_classification_template.json`: 라벨 템플릿 예시

## 3. 환경 설정
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

권장 환경:
- Python 3.10 이상
- CUDA 사용 가능한 PyTorch 환경(임베딩/학습용)

## 4. 데이터 구조 예시
패치 디렉토리 예시:
```text
/path/to/40x_patch/
  C228T/
    TC_XX_0001/
      tile_x123_y456.png
  C250T/
  Wild/
```

임베딩 출력 예시(`preprocess_data.py` 결과):
```text
/path/to/embedding/
  C228T/
    npy/TC_XX_0001.npy
    json/TC_XX_0001.json
  C250T/
  Wild/
```

## 5. 임베딩 생성
단일 서브타입 실행:
```bash
python src/data/preprocess_data.py \
  --tile_dir /path/to/40x_patch/C228T \
  --out_dir /path/to/embedding/C228T \
  --batch_size 512
```

전체 서브타입 병렬 실행:
```bash
cd src/data
TILE_BASE=/path/to/40x_patch \
OUT_BASE=/path/to/embedding \
GPU_C228T=0 GPU_C250T=1 GPU_WILD=2 \
bash run_embedding.sh
```

## 6. CV Split 생성
```bash
python src/data/create_cv_splits.py \
  --data_root /path/to/embedding \
  --label_file /path/to/Thyroid_TERT_labels.xlsx \
  --output_dir /path/to/splits \
  --n_splits 5 \
  --seed 42 \
  --train_ratio 0.7 \
  --val_ratio 0.1 \
  --test_ratio 0.2
```

`cv_splits` JSON에는 `folds[*].train_wsis`, `val_wsis`, `test_wsis`에 전체 `.npy` 경로가 들어가야 합니다.

## 7. 모델 학습
기본 실행:
```bash
cd src/training
bash train.sh
```

파라미터 오버라이드 예시:
```bash
cd src/training
CV_SPLIT_FILE=/path/to/cv_splits_tert.json \
MODEL_SAVE_DIR=/path/to/outputs/thyroid_tert_v0.1.0 \
MODEL_TYPE=transmil \
CUDA_VISIBLE_DEVICES=0 \
EPOCHS=100 \
LR=1e-4 \
BAG_SIZE=5000 \
bash train.sh
```

참고:
- 데이터 누수 검사(train/val/test 중복)가 기본 적용되며, 중복 검출 시 학습이 중단됩니다.
- 핵심 결과 파일: `results_cv_summary_optimal.json`

## 8. 주요 산출물
학습 완료 후 `MODEL_SAVE_DIR`에 생성:
- `results_cv_summary_optimal.json`
- `attention_scores/attention_scores_fold*.json`
- `checkpoints/*.pt` (`--save_model` 사용 시)
- `visualizations/` (`--generate_plots` 사용 시)
- `heatmaps/` (best fold attention heatmap)

## 9. MLflow 연동 (선택)
학습 전 환경변수 설정:
```bash
export MLFLOW_TRACKING_URI="https://your-mlflow-server"
export MLFLOW_EXPERIMENT_NAME="thyroid_tert"
export MLFLOW_TRACKING_INSECURE_TLS="false"
```

모델 등록 이름(선택):
```bash
export MLFLOW_REGISTERED_MODEL_NAME="thyr-tert"
```

## 10. 보안 주의사항
이 공개용 패키지는 내부 호스트/IP/절대경로를 제거하고 플레이스홀더 경로만 유지합니다.
아래 항목은 커밋하지 마세요:
- 비공개 원본 데이터셋
- 내부 인증정보/토큰
- 내부 추적 서버 주소
