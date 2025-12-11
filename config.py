import os
from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class Config:
    # Paths
    PROJECT_ROOT: str = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR: str = os.path.join(PROJECT_ROOT, "data")
    FEATURES_DIR: str = os.path.join(DATA_DIR, "features")
    # TODO: Verify correct annotation path structure
    ANNOTATIONS_DIR: str = os.path.join(
        DATA_DIR, 
        "annotations", 
        "annotations averaged per song", 
        "dynamic (per second annotations)"
    )
    LOG_DIR: str = os.path.join(PROJECT_ROOT, "logs")
    SAVED_MODELS_DIR: str = os.path.join(PROJECT_ROOT, "saved_models")
    
    # System & Logging
    DEVICE: str = "cuda" # 'cuda' or 'cpu'
    LOG_TO_FILE: bool = True
    LOG_FILE_NAME: str = "training.log"
    
    # Data Processing
    SAMPLING_RATE_HZ: int = 2  # Features are 2Hz (500ms windows)
    SEQUENCE_LENGTH: int = 59 # Explicitly requested 59 time steps
    BATCH_SIZE: int = 32
    TEST_SPLIT: float = 0.2
    VAL_SPLIT: float = 0.1
    RANDOM_SEED: int = 42
    
    # Model Hyperparameters
    # TODO: Refine architecture based on experiments
    NB_FILTERS: int = 64
    KERNEL_SIZE: int = 5
    NB_STACKS: int = 3 # 3 blocks
    DILATIONS: Tuple[int, ...] = (1, 1, 1) # No dilation
    DROPOUT_RATE: float = 0.1
    USE_SKIP_CONNECTIONS: bool = True
    RETURN_SEQUENCES: bool = True 
    OUTPUT_DIM: int = 2 # Valence, Arousal
    
    # Training
    LEARNING_RATE: float = 0.001
    EPOCHS: int = 100
    PATIENCE: int = 10 # For EarlyStopping

    def __post_init__(self):
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.SAVED_MODELS_DIR, exist_ok=True)

config = Config()
