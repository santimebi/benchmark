"""
utils/hp_spaces.py
───────────────────────────────────────────────
Definición de los espacios de búsqueda y objetivos para la optimización de
hiperparámetros (Optuna).
"""

from pathlib import Path
from typing import Dict, Any
import optuna
from utils.config import MODELS_PATH

HP_SPACES: Dict[str, Dict[str, Any]] = {
    "standard": {
        "suggest_fn": lambda trial, model_arch="": {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [8, 16, 32, 64]) if "resnet" not in str(model_arch).lower() else 512,
            "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "epochs": 300,
        },
        "objective_type": "val_loss",
        "output_path": MODELS_PATH / "best_hp.json",
    },
    "cfk": {
        "suggest_fn": lambda trial, model_arch="": {
            "lr": trial.suggest_float("lr", 1e-5, 1e-1, log=True),
            "k": trial.suggest_categorical("k", [1, 5, 9, 13, 17, 18]) if "resnet" in str(model_arch).lower() else trial.suggest_int("k", 1, 3),
            "epochs": 20,  # Fijo a 20 épocas para el desaprendizaje
        },
        "objective_type": "unlearning_loss",
        "output_path": MODELS_PATH / "best_cfk_hp.json",
    },
    "euk": {
        "suggest_fn": lambda trial, model_arch="": {
            "lr": trial.suggest_float("lr", 1e-5, 1e-1, log=True),
            "k": trial.suggest_categorical("k", [1, 5, 9, 13, 17, 18]) if "resnet" in str(model_arch).lower() else trial.suggest_int("k", 1, 3),
            "epochs": 20,  # Fijo a 20 épocas para el desaprendizaje
        },
        "objective_type": "unlearning_loss",
        "output_path": MODELS_PATH / "best_euk_hp.json",
    },
    "cfgk": {
        "suggest_fn": lambda trial, model_arch="": {
            "lr": trial.suggest_float("lr", 1e-5, 1e-1, log=True),
            "k": trial.suggest_categorical("k", [1, 5, 9, 13, 17, 18]) if "resnet" in str(model_arch).lower() else trial.suggest_int("k", 1, 3),
            "c": trial.suggest_float("c", -2.0, 8.0),
            "gamma": trial.suggest_float("gamma", -1.0, 10.0),
            "epochs": 10,
        },
        "objective_type": "unlearning_loss",
        "output_path": MODELS_PATH / "best_cfgk_hp.json",
    },
    "rurk": {
        "suggest_fn": lambda trial, model_arch="": {
            "lr": trial.suggest_categorical("lr", [0.01]),
            "momentum": trial.suggest_categorical("momentum", [0.90]),
            "weight_decay": trial.suggest_categorical("weight_decay", [5e-4]),
            "tau": trial.suggest_categorical("tau", [0.03]),
            "lambda_f": trial.suggest_categorical("lambda_f", [0.03]),
            "lambda_a": trial.suggest_categorical("lambda_a", [0.00045]),
            "epochs": trial.suggest_categorical("epochs", [2]),
            "max_total_iterations": trial.suggest_categorical("max_total_iterations", [200]),
            "gradient_clip_norm": trial.suggest_categorical("gradient_clip_norm", [1.0]),
            "v": trial.suggest_categorical("v", [1]),
            "batch_size": trial.suggest_categorical("batch_size", [128])
        },
        "objective_type": "unlearning_loss",
        "output_path": MODELS_PATH / "best_rurk_hp.json",
    }
}

