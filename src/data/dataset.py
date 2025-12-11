import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Optional
import pickle
import os

from config import config
from src.data.loader import DataLoader

class TCNDataPipeline:
    def __init__(self, emotion_type: str = 'arousal', limit: int = None):
        self.config = config
        self.loader = DataLoader()
        self.emotion_type = emotion_type
        self.limit = limit
        self.scaler = StandardScaler()
        
    def _split_data(self, X: list, y: list) -> Tuple[list, list, list, list, list, list]:
        """
        Splits data into train, validation, and test sets.
        Splitting is done at the song level to avoid data leakage.
        """
        # First split: Train+Val vs Test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=self.config.TEST_SPLIT, random_state=self.config.RANDOM_SEED
        )
        
        # Second split: Train vs Val (from Train+Val)
        # Adjust val_split relative to the remaining data
        relative_val_split = self.config.VAL_SPLIT / (1 - self.config.TEST_SPLIT)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=relative_val_split, random_state=self.config.RANDOM_SEED
        )
        
        return X_train, y_train, X_val, y_val, X_test, y_test

    def _fit_transform_scaler(self, X_train: list) -> list:
        # Concatenate all training data to fit the scaler
        flat_X_train = np.vstack(X_train)
        self.scaler.fit(flat_X_train)
        
        # Transform training data
        X_scaled = [self.scaler.transform(x) for x in X_train]
        return [np.nan_to_num(x) for x in X_scaled]

    def _transform_scaler(self, X: list) -> list:
        X_scaled = [self.scaler.transform(x) for x in X]
        return [np.nan_to_num(x) for x in X_scaled]

    def _prepare_sequences(self, X: list, y: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts lists of variable-length songs into fixed-length sequences.
        Uses a sliding window approach.
        """
        seq_len = self.config.SEQUENCE_LENGTH
        X_seqs = []
        y_seqs = []
        
        for song_feats, song_labels in zip(X, y):
            # TODO: Optimize sequence generation (e.g. using tf.data.Dataset.window)
            if len(song_feats) < seq_len:
                continue
            
            # TODO: Implement sequence generation logic here if moving away from generator
            
            num_sequences = len(song_feats) - seq_len + 1
            if num_sequences <= 0: continue

            # TODO: Consider optimization for large datasets if RAM becomes an issue
            pass

        return np.array(X_seqs), np.array(y_seqs)

    def generator(self, X, y):
        """
        Generator for tf.data.Dataset
        Yields (sequence, target_sequence)
        """
        seq_len = self.config.SEQUENCE_LENGTH
        
        for song_feats, song_labels in zip(X, y):
            num_samples = len(song_feats)
            if num_samples < seq_len:
                continue
            
            # Yield sliding windows
            for i in range(num_samples - seq_len + 1):
                x_window = song_feats[i : i + seq_len]
                y_window = song_labels[i : i + seq_len] # Sequence to sequence
                
                # TODO: Ensure reshape matches output dim logic
                if y_window.ndim == 1:
                    y_window = y_window.reshape(-1, 1)
                
                yield x_window, y_window

    def create_dataset(self, X: list, y: list, training: bool = True) -> tf.data.Dataset:
        """
        Creates a tf.data.Dataset from list of arrays.
        """
        # Determine feature dim
        feature_dim = X[0].shape[1]
        
        output_signature = (
            tf.TensorSpec(shape=(self.config.SEQUENCE_LENGTH, feature_dim), dtype=tf.float32),
            tf.TensorSpec(shape=(self.config.SEQUENCE_LENGTH, self.config.OUTPUT_DIM), dtype=tf.float32)
        )
        
        dataset = tf.data.Dataset.from_generator(
            lambda: self.generator(X, y),
            output_signature=output_signature
        )
        
        if training:
            dataset = dataset.shuffle(buffer_size=10000)
            
        dataset = dataset.batch(self.config.BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset

    def get_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Main method to get train, val, test datasets.
        """
        # Load raw data
        raw_X, raw_y = self.loader.load_dataset(self.emotion_type, limit=self.limit)
        
        # Split
        X_train, y_train, X_val, y_val, X_test, y_test = self._split_data(raw_X, raw_y)
        
        # Normalize (Fit on train, apply to all)
        X_train = self._fit_transform_scaler(X_train)
        X_val = self._transform_scaler(X_val)
        X_test = self._transform_scaler(X_test)
        
        # Create Datasets
        train_ds = self.create_dataset(X_train, y_train, training=True)
        val_ds = self.create_dataset(X_val, y_val, training=False)
        test_ds = self.create_dataset(X_test, y_test, training=False)
        
        return train_ds, val_ds, test_ds

    def save_scaler(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
            
    def load_scaler(self, path: str):
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)

if __name__ == "__main__":
    # Test
    pipeline = TCNDataPipeline(limit=10)
    train_ds, val_ds, test_ds = pipeline.get_datasets()
    
    print("Datasets created.")
    for x, y in train_ds.take(1):
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {y.shape}")
