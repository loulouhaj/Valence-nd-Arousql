import os
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
import re
from config import config

class DataLoader:
    def __init__(self):
        self.features_dir = config.FEATURES_DIR
        self.annotations_dir = config.ANNOTATIONS_DIR
        self.sampling_rate = config.SAMPLING_RATE_HZ
        self.features_start_time_ms = 0 # Features start at 0
        self.annotations_start_time_ms = 15000 # Annotations start at 15000ms

    def load_annotations(self, type: str = 'arousal') -> pd.DataFrame:
        """
        Loads annotations for 'arousal' or 'valence'.
        Returns a DataFrame indexed by song_id, with columns as sample timestamps.
        """
        file_path = os.path.join(self.annotations_dir, f"{type}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Annotation file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        df = df.set_index('song_id')
        return df

    def get_feature_path(self, song_id: int) -> str:
        return os.path.join(self.features_dir, f"{song_id}.csv")

    def load_song_features(self, song_id: int) -> pd.DataFrame:
        """
        Loads features for a specific song.
        """
        feature_path = self.get_feature_path(song_id)
        if not os.path.exists(feature_path):
            print(f"Warning: Feature file for song {song_id} not found.")
            return None
        
        # Features are semicolon separated
        df = pd.read_csv(feature_path, sep=';')
        return df.fillna(0)

    def align_data(self, features: pd.DataFrame, annotations: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aligns features and annotations based on timestamps.
        Annotations columns are like 'sample_15000ms'.
        Features have 'frameTime' in seconds.
        """
        # TODO: Validate annotation index matches expected format
        
        # Filter annotations to keep only sample_Xms columns
        valid_indices = [c for c in annotations.index if str(c).startswith('sample_')]
        filtered_annotations = annotations[valid_indices]
        
        # Extract times from annotation keys
        # e.g. sample_15000ms -> 15000
        annot_times = [int(re.search(r'sample_(\d+)ms', c).group(1)) for c in filtered_annotations.index]
        annot_times = np.array(annot_times)
        
        # Extract labels
        labels = filtered_annotations.values.astype(float)
        
        # Check for NaNs in labels
        valid_mask = ~np.isnan(labels)
        annot_times = annot_times[valid_mask]
        labels = labels[valid_mask]
        
        # Prepare features
        # frameTime is in seconds, convert to ms
        feature_times = (features['frameTime'] * 1000).astype(int).values
        
        # We need to match features to annotation times.
        if len(annot_times) > 1:
            interval = annot_times[1] - annot_times[0]
            if interval != 500:
                 # TODO: Verify interval consistency
                 pass
        
        # Alignment:       
        aligned_features = []
        aligned_labels = []
        
        for t, label in zip(annot_times, labels):
            # Find closest frame
            idx = np.argmin(np.abs(feature_times - t))
            if abs(feature_times[idx] - t) < 250: 
                feat_row = features.iloc[idx].drop('frameTime').values.astype(float)
                aligned_features.append(feat_row)
                aligned_labels.append(label)
            else:
                # TODO: Handle missing features for timestamps
                pass
                
        return np.array(aligned_features), np.array(aligned_labels)

    def load_dataset(self, emotion_type='arousal', limit: int = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Loads the entire dataset.
        If emotion_type is 'both', returns labels as (Time, 2) [arousal, valence].
        """
        if emotion_type == 'both':
            arousal_df = self.load_annotations('arousal')
            valence_df = self.load_annotations('valence')
            # Use intersection of indices
            common_ids = arousal_df.index.intersection(valence_df.index)
            annot_df = arousal_df.loc[common_ids] # Use arousal as base for iteration
        else:
            annot_df = self.load_annotations(emotion_type)
            
        X = []
        y = []
        
        print(f"Loading {emotion_type} dataset...")
        processed_count = 0
        for song_id, row in annot_df.iterrows():
            if limit and processed_count >= limit:
                break
            
            features_df = self.load_song_features(song_id)
            if features_df is None:
                continue
            
            if emotion_type == 'both':
                # Load both
                arousal_row = arousal_df.loc[song_id]
                valence_row = valence_df.loc[song_id]
                
                feats, arousal_labs = self.align_data(features_df, arousal_row)
                _, valence_labs = self.align_data(features_df, valence_row)
                
                if len(feats) > 0 and len(feats) == len(arousal_labs) == len(valence_labs):
                    # TODO: Verify label stacking order [valence, arousal] matches model output
                    labs = np.stack([valence_labs, arousal_labs], axis=1)
                    
                    X.append(feats)
                    y.append(labs)
                    processed_count += 1
                else:
                    # TODO: specific error logging for mismatch
                    pass
            else:
                feats, labs = self.align_data(features_df, row)
                
                if len(feats) > 0 and len(feats) == len(labs):
                    X.append(feats)
                    y.append(labs)
                    processed_count += 1
                else:
                    print(f"Skipping song {song_id}: No overlapping data.")
                
        return X, y

if __name__ == "__main__":
    # Simple test
    loader = DataLoader()
    X, y = loader.load_dataset('arousal')
    print(f"Loaded {len(X)} songs.")
    if len(X) > 0:
        print(f"Shape of first song features: {X[0].shape}")
        print(f"Shape of first song labels: {y[0].shape}")
