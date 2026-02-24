# -*- coding: utf-8 -*-
"""
Thyroid TERT Mutation Main Training Script

Task: TERT Promoter Mutation Prediction (Wild vs Mutant)
- Wild (no mutation) = 0
- Mutant (C228T or C250T) = 1

Workflow:
    1. Parse command-line arguments
    2. Run 5-fold cross-validation training (train_tert.py)

Entry Point:
    Main entry point for thyroid_tert_v0.1.0 training pipeline.
"""

import os
import sys

# GPU 설정을 가장 먼저 해야 함 (torch import 전에)
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print(f"[main.py] CUDA_VISIBLE_DEVICES not set, forcing GPU 0")
else:
    print(f"[main.py] CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

import argparse

# =========================
# Path Configuration
# =========================
# Add src directory to Python path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

from train_tert import run_training
from mlflow_utils import upload_to_mlflow


def main():
    """
    Main entry point for TERT training.

    Steps:
    1. Parse arguments
    2. Run training (5-fold CV)
    """

    # Parse arguments
    parser = argparse.ArgumentParser(description='Thyroid TERT: Train MIL model (ABMIL/TransMIL) for TERT mutation prediction (Wild vs Mutant)')
    parser.add_argument('--model_save_dir', type=str, required=True,
                        help='Directory to save model checkpoints and results')
    parser.add_argument('--cv_split_file', type=str, required=True,
                        help='Path to CV split JSON file')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--bag_size', type=int, default=2000,
                        help='Bag size for MIL (default: 2000)')
    parser.add_argument('--model_type', type=str, default='abmil', choices=['abmil', 'transmil'],
                        help='Model architecture to train (default: abmil)')
    parser.add_argument('--in_dim', type=int, default=1536,
                        help='Input feature dimension (default: 1536)')
    parser.add_argument('--dropout', type=float, default=0.25,
                        help='Dropout used by model blocks (default: 0.25)')

    # ABMIL params
    parser.add_argument('--embed_dim', type=int, default=512,
                        help='ABMIL embedding dimension (default: 512)')
    parser.add_argument('--attn_dim', type=int, default=384,
                        help='ABMIL attention hidden dimension (default: 384)')
    parser.add_argument('--num_fc_layers', type=int, default=2,
                        help='ABMIL patch-embed MLP depth (default: 2)')

    # TransMIL params (GitHub default style)
    parser.add_argument('--transmil_embed_dim', type=int, default=512,
                        help='TransMIL embedding dimension (default: 512)')
    parser.add_argument('--transmil_num_heads', type=int, default=8,
                        help='TransMIL attention heads (default: 8)')
    parser.add_argument('--transmil_num_layers', type=int, default=2,
                        help='TransMIL transformer layers (default: 2)')
    parser.add_argument('--transmil_num_landmarks', type=int, default=256,
                        help='Nyström landmarks for TransMIL attention (default: 256)')
    parser.add_argument('--transmil_pinv_iterations', type=int, default=6,
                        help='Nyström pseudo-inverse iterations (default: 6)')
    parser.add_argument('--disable_layer_norm', action='store_true',
                        help='Disable input layer normalization before patch embedding')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--save_model', action='store_true',
                        help='Save model checkpoints')
    parser.add_argument('--save_best_only', action='store_true',
                        help='Save only best model per fold')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--generate_plots', action='store_true',
                        help='Generate visualization plots')
    parser.add_argument('--test_fold', type=int, default=None,
                        help='Test only specific fold (default: None, test all folds)')

    args = parser.parse_args()

    # Run training
    print(f"{'='*80}")
    print(f"Starting TERT Mutation Prediction Training Pipeline")
    print(f"Task: Wild (0) vs Mutant (1)")
    print(f"Model: {args.model_type}")
    print(f"{'='*80}\n")

    training_results = run_training(args)

    # Check if MLflow upload should be performed
    if training_results and training_results.get('model_checkpoint_path') and args.save_model:
        print(f"\n{'='*80}")
        print(f"Uploading to MLflow")
        print(f"{'='*80}\n")

        try:
            upload_to_mlflow(
                model_save_dir=training_results['model_save_dir'],
                json_path=training_results['json_path'],
                model_checkpoint_path=training_results['model_checkpoint_path'],
                lr=args.lr,
                epochs=args.epochs,
                bag_size=args.bag_size,
                seed=args.seed,
                model_type=args.model_type,
            )
            print(f"\n[+] MLflow upload completed!")

        except Exception as e:
            print(f"\n[!] MLflow upload failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n[i] Skipping MLflow upload (no model checkpoint or --save_model not specified)")

    print(f"\n{'='*80}")
    print(f"Pipeline Completed!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
