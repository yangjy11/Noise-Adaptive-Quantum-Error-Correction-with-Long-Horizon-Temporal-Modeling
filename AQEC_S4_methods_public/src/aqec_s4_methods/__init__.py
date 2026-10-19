"""AQEC-S4 learning methods, independent of the private simulation environment."""

from .config import ModelConfig
from .data import Episode, TaskBatch, make_synthetic_episodes, read_task_batch
from .losses import FeedbackGRAPELoss, SyntheticComplexMSELoss
from .meta import AdaptiveInnerLoopOptimizer, FOMAMLTrainer
from .model import QuantumS4Model, S4Block, S4Layer, hippo_initialization
from .transfer import finetune, parameter_groups
from .checkpoints import load_checkpoint, save_checkpoint

__version__ = "0.1.0"
__all__ = [
    "ModelConfig", "Episode", "TaskBatch", "make_synthetic_episodes", "read_task_batch",
    "FeedbackGRAPELoss", "SyntheticComplexMSELoss", "AdaptiveInnerLoopOptimizer",
    "FOMAMLTrainer", "QuantumS4Model", "S4Block", "S4Layer", "hippo_initialization",
    "finetune", "parameter_groups", "load_checkpoint", "save_checkpoint",
]
