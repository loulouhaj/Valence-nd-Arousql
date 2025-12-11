import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
from config import config
from src.utils.logger import logger

class ModelEvaluator:
    def __init__(self, model: tf.keras.Model):
        self.model = model
        self.config = config

    def evaluate(self, test_ds: tf.data.Dataset):
        """
        Evaluates the model on the test dataset.
        """
        logger.info("Starting evaluation...")
        
        # Get predictions
        y_true = []
        y_pred = []
        
        for batch_x, batch_y in test_ds:
            preds = self.model.predict(batch_x, verbose=0)
            
            # TODO: verify TCN output shape logic with current config
            if not self.config.RETURN_SEQUENCES:
                 # TODO: handle many-to-one logic if used
                 ground_truth = batch_y[:, -1, :]
            else:
                 ground_truth = batch_y
            
            y_true.append(ground_truth.numpy())
            y_pred.append(preds)
            
        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred)
        
        # Flatten for metrics
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Metrics
        mse = mean_squared_error(y_true_flat, y_pred_flat)
        mae = mean_absolute_error(y_true_flat, y_pred_flat)
        r2 = r2_score(y_true_flat, y_pred_flat)
        
        logger.info("Evaluation metrics:")
        logger.info(f"MSE: {mse:.4f}")
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"R2 Score: {r2:.4f}")
        
        return {
            "mse": mse,
            "mae": mae,
            "r2": r2
        }

    def plot_predictions(self, y_true, y_pred, title="Predictions vs True"):
        # Helper to plot separate if needed
        # Assuming run manually for now
        pass
