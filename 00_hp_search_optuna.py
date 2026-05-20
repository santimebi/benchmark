"""
00_hp_search_optuna.py
─────────────────────────────────────────────────────────────
Búsqueda de hiperparámetros con Optuna para el modelo base (BaseMLP).
Se ejecuta ANTES de 3_train_model.py para determinar la
mejor combinación de: hidden_dim, learning_rate, batch_size y epochs.

Los mejores hiperparámetros se guardan en
    models/best_hp.json
para que 3_train_model.py pueda consumirlos directamente.
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

import optuna
from optuna.exceptions import TrialPruned

from utils.config import DATASETS_PATH
from models.base_nn import BaseMLP

# ──────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────
SEED = 0                       # Seed del split a utilizar para la búsqueda
N_TRIALS = 50                  # Número de trials de Optuna
INPUT_DIM = 2                  # Dimensión de entrada del espiral
OUTPUT_DIM = 3                 # Número de clases
EPOCHS = 100                   # Épocas para el entrenamiento
BEST_HP_PATH = Path("models/best_hp.json")


def load_data(seed: int):
    """
    Carga los datos del split indicado y devuelve tensores de train y validación.

    Combina Retain + Forget como conjunto de entrenamiento (modelo base
    original) y devuelve el conjunto de validación por separado.

    Args:
        seed: Semilla del split a cargar.

    Returns:
        tuple: ``(X_train_t, y_train_t, X_val_t, y_val_t)`` como tensores PyTorch.

    Raises:
        FileNotFoundError: Si el ``.npz`` del split no existe.
    """
    data_path = DATASETS_PATH / f"spiral_splits_seed_{seed}.npz"
    if not data_path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset para la seed {seed}. "
            "Ejecuta 2_split_dataset.py primero."
        )

    data = np.load(data_path)

    # Train = Retain + Forget (modelo base original)
    X_train = np.vstack([data["X_retain"], data["X_forget"]])
    y_train = np.concatenate([data["y_retain"], data["y_forget"]])

    X_val = data["X_val"]
    y_val = data["y_val"]

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    return X_train_t, y_train_t, X_val_t, y_val_t


def objective(trial: optuna.Trial) -> float:
    """
    Función objetivo que Optuna minimiza.

    Define el espacio de búsqueda (``hidden_dim``, ``lr``, ``batch_size``,
    ``epochs``), entrena un ``BaseMLP`` y devuelve la validation loss
    del último epoch. Reporta valores intermedios para pruning.

    Args:
        trial: Objeto Trial de Optuna que sugiere los hiperparámetros.

    Returns:
        float: Validation loss del último epoch completado.

    Raises:
        TrialPruned: Si Optuna decide podar el trial por bajo rendimiento.
    """
    # ── Espacio de búsqueda ──
    hidden_dim = trial.suggest_categorical("hidden_dim", [8, 16, 32, 64])
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    epochs = trial.suggest_int("epochs", 50, 300, step=50)

    # ── Datos ──
    X_train_t, y_train_t, X_val_t, y_val_t = load_data(SEED)
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # ── Modelo ──
    model = BaseMLP(input_dim=INPUT_DIM, hidden_dim=hidden_dim, output_dim=OUTPUT_DIM)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ── Entrenamiento ──
    for epoch in range(EPOCHS):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Validación cada epoch para reporting y pruning
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()

        # Reportar valor intermedio a Optuna para pruning temprano
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise TrialPruned()

    return val_loss


def run_search(n_trials: int = N_TRIALS) -> dict:
    """
    Ejecuta la búsqueda de hiperparámetros y guarda los mejores en JSON.

    Crea un estudio Optuna con ``MedianPruner``, ejecuta ``n_trials``
    optimizaciones, imprime los resultados y persiste los mejores
    hiperparámetros en ``BEST_HP_PATH``.

    Args:
        n_trials: Número de trials a ejecutar.

    Returns:
        dict: Diccionario con los mejores hiperparámetros encontrados.
    """
    study = optuna.create_study(
        direction="minimize",
        study_name="base_mlp_hp_search",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20),
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # ── Resultados ──
    best = study.best_trial
    print("\n" + "=" * 60)
    print("  MEJORES HIPERPARÁMETROS ENCONTRADOS")
    print("=" * 60)
    print(f"  Val Loss:   {best.value:.6f}")
    for key, value in best.params.items():
        print(f"  {key:12s}: {value}")
    print("=" * 60)

    # ── Guardar a JSON ──
    BEST_HP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BEST_HP_PATH, "w", encoding="utf-8") as f:
        json.dump(best.params, f, indent=2, ensure_ascii=False)
    print(f"\nHiperparámetros guardados en: {BEST_HP_PATH}")

    return best.params


if __name__ == "__main__":
    run_search()
