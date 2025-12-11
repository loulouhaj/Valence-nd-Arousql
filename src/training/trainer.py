import tensorflow as tf
import os
from config import config
from src.utils.logger import logger, get_tensorboard_log_dir

class ModelTrainer:
    def __init__(self, model: tf.keras.Model):
        self.model = model
        self.config = config
        
    def compile(self):
        """
        Compiles the model with optimizer and loss function.
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE)
        
        # TODO: verify metrics suitable for dual output
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        logger.info("Model compiled.")

    def fit(self, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset):
        """
        Runs the training loop.
        """
        logger.info("Starting training...")
        
        # Callbacks
        callbacks = self._get_callbacks()
        
        history = self.model.fit(
            train_ds,
            epochs=self.config.EPOCHS,
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Training completed.")
        return history

    def _get_callbacks(self) -> list:
        # TODO: configure callback list dynamically
        callbacks = []
        
        # TensorBoard
        log_dir = get_tensorboard_log_dir()
        logger.info(f"TensorBoard log directory: {log_dir}")
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir))
        
        # Model Checkpoint
        checkpoint_path = os.path.join(self.config.SAVED_MODELS_DIR, "tcn_best.keras")
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ))
        
        # Early Stopping
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config.PATIENCE,
            restore_best_weights=True,
            verbose=1
        ))
        
        return callbacks
