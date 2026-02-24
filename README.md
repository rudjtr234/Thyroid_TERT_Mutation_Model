# Thyroid TERT Promoter Mutation Prediction (Public)

MIL-based training pipeline for predicting TERT promoter mutation from WSI patch embeddings.

## Task
- Binary classification: `Wild = 0`, `Mutant(C228T/C250T) = 1`
- Models: `ABMIL`, `TransMIL`

## Project Layout
- `src/training/main.py`: entrypoint
- `src/training/train_tert.py`: 5-fold CV training/eval
- `src/training/mlflow_utils.py`: optional MLflow logging (env-driven)
- `src/training/register_model.py`: optional model registry utility
- `src/data/`: preprocessing, dataset, split generation
- `src/evaluation/`: metrics and visualization
- `src/models/`: ABMIL / TransMIL implementations
- `config/cv_splits_tert_example.json`: sample CV split template

## Security Notes
This public package excludes private dataset paths, sample manifests, and internal server addresses.
Set environment variables instead:
- `MLFLOW_TRACKING_URI`
- `MLFLOW_EXPERIMENT_NAME` (optional, default: `thyroid_tert`)
- `MLFLOW_TRACKING_INSECURE_TLS` (optional)
- `MLFLOW_REGISTERED_MODEL_NAME` (optional, for `register_model.py`)

## Quick Start
```bash
cd src/training
bash train.sh
```

Override defaults as needed:
```bash
CV_SPLIT_FILE=/path/to/cv_splits.json \
MODEL_SAVE_DIR=/path/to/output \
CUDA_VISIBLE_DEVICES=0 \
bash train.sh
```
