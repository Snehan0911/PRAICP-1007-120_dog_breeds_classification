"""
CanineVision AI — CLI Model Training Script.

Usage:
    python scripts/train_model.py --model mobilenet_v2 --epochs1 5 --epochs2 3 --batch-size 32
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.trainer import ModelTrainer
from src.utils.logger import get_logger
from src.utils.paths import ProjectPaths
from src.utils.seed import set_all_seeds

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="CanineVision AI — Model Training CLI")
    parser.add_argument(
        "--model",
        type=str,
        default="mobilenet_v2",
        choices=["mobilenet_v2", "custom_cnn", "efficientnet_b0"],
        help="Architecture to train",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--epochs1", type=int, default=5, help="Phase 1 epochs")
    parser.add_argument("--epochs2", type=int, default=3, help="Phase 2 fine-tuning epochs")
    parser.add_argument("--lr1", type=float, default=0.001, help="Phase 1 learning rate")
    parser.add_argument("--lr2", type=float, default=0.0001, help="Phase 2 learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    set_all_seeds(args.seed)
    paths = ProjectPaths()
    paths.ensure_all_exist()

    logger.info("Initializing trainer for %s...", args.model)
    trainer = ModelTrainer(
        model_name=args.model,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    trainer.train(
        splits_dir=paths.data_splits,
        epochs_phase1=args.epochs1,
        epochs_phase2=args.epochs2,
        lr_phase1=args.lr1,
        lr_phase2=args.lr2,
        checkpoint_dir=paths.artifacts_models,
        history_dir=paths.outputs_training,
    )


if __name__ == "__main__":
    main()
