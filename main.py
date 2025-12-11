import argparse
import sys
import tensorflow as tf
from config import config
from src.data.dataset import TCNDataPipeline
from src.models.tcn_model import build_tcn_model
from src.training.trainer import ModelTrainer
from src.evaluation.evaluator import ModelEvaluator
from src.utils.logger import logger

def main():
    parser = argparse.ArgumentParser(description="TCN Training for DEAM Dataset")
    parser.add_argument('--limit', type=int, default=None, help='Limit number of songs for debugging')
    parser.add_argument('--epochs', type=int, default=None, help='Override number of epochs')
    parser.add_argument('--emotion', type=str, default='both', choices=['arousal', 'valence', 'both'], help='Target emotion (or "both")')
    
    args = parser.parse_args()
    
    if args.epochs:
        config.EPOCHS = args.epochs
    
    logger.info(f"Starting pipeline for {args.emotion}...")
    
    # 1. Data
    logger.info("Initializing Data Pipeline...")
    data_pipeline = TCNDataPipeline(emotion_type=args.emotion, limit=args.limit)
    train_ds, val_ds, test_ds = data_pipeline.get_datasets()
    
    if train_ds is None:
        logger.error("Failed to create datasets.")
        sys.exit(1)

    # Determine input shape from dataset
    for x, y in train_ds.take(1):
        input_shape = x.shape[1:] # (Sequence_Length, Features)
        logger.info(f"Input shape determined: {input_shape}")
        break
        
    # 2. Model
    logger.info("Building Model...")
    model = build_tcn_model(input_shape)
    model.summary(print_fn=logger.info)
    
    # 3. Training
    logger.info("Initializing Trainer...")
    trainer = ModelTrainer(model)
    trainer.compile()
    
    history = trainer.fit(train_ds, val_ds)
    
    # 4. Evaluation
    logger.info("Evaluating on Test Set...")
    evaluator = ModelEvaluator(model)
    metrics = evaluator.evaluate(test_ds)
    
    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
