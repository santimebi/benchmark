"""
00_hp_search.py
───────────────────────────────────────────────
Búsqueda de hiperparámetros generalizada con Optuna.
Soporta optimización para:
  - 'standard': Hiperparámetros de entrenamiento del modelo base/naive.
  - 'cfk': Hiperparámetros del protocolo de desentrenamiento CFK.

Guarda los mejores hiperparámetros en el archivo JSON especificado en utils/hp_spaces.py.
"""

import argparse
import json
import inspect
import importlib
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import optuna

from utils.config import DATASETS_PATH, MODELS_PATH
from utils.hp_spaces import HP_SPACES
from utils.protocols import get_protocol

WEIGHTS_DIR = MODELS_PATH / "weights"



def load_class(class_path: str):
    """
    Carga dinámicamente una clase a partir de su string de importación.
    """
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except Exception as e:
        raise ImportError(f"No se pudo cargar la clase '{class_path}': {e}")


def load_best_hp(hp_file_path: Path) -> dict:
    """
    Carga los mejores hiperparámetros desde un archivo JSON si existe.
    """
    if not hp_file_path.exists():
        print(f"[WARN] No se encontró {hp_file_path}. Usando hidden_dim=64 y batch_size=16 por defecto.")
        return {}
    with open(hp_file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def objective(trial: optuna.Trial, protocol: str, seed: int, model_arch: str, dataset: str = "spiral") -> float:
    """
    Función objetivo a minimizar por Optuna.
    """
    if protocol not in HP_SPACES:
        raise ValueError(f"Protocolo '{protocol}' no configurado en HP_SPACES.")
        
    hp_config = HP_SPACES[protocol]
    suggested_hp = hp_config["suggest_fn"](trial, model_arch=model_arch)
    objective_type = hp_config["objective_type"]
    
    if objective_type == "val_loss":
        # ──────────────────────────────────────────────
        # Optimización para entrenamiento estándar (Base Model)
        # ──────────────────────────────────────────────
        hidden_dim = suggested_hp.get("hidden_dim", 16)
        lr = suggested_hp.get("lr", 1e-3)
        batch_size = suggested_hp.get("batch_size", 32)
        epochs = suggested_hp.get("epochs", 150)
        
        data_path = DATASETS_PATH / f"{dataset}_splits_seed_{seed}.npz"
        if not data_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo de datos {data_path}. Ejecuta 2_split_dataset.py primero.")
            
        data = np.load(data_path)
        X_train = np.vstack([data["X_retain"], data["X_forget"]])
        y_train = np.concatenate([data["y_retain"], data["y_forget"]])
        X_val = data["X_val"]
        y_val = data["y_val"]
        
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.long)
        
        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)
        
        model_class = load_class(model_arch)
        num_features = X_train.shape[1] if len(X_train.shape) == 2 else int(np.prod(X_train.shape[1:]))
        num_classes = int(np.max(np.concatenate([y_train, y_val])) + 1)
        
        sig = inspect.signature(model_class.__init__)
        model_kwargs = {}
        if "input_dim" in sig.parameters:
            model_kwargs["input_dim"] = num_features
        elif "in_dim" in sig.parameters:
            model_kwargs["in_dim"] = num_features
        elif "in_features" in sig.parameters:
            model_kwargs["in_features"] = num_features
        elif "in_channels" in sig.parameters:
            model_kwargs["in_channels"] = X_train.shape[1]
            
        if "hidden_dim" in sig.parameters:
            model_kwargs["hidden_dim"] = hidden_dim
            
        if "output_dim" in sig.parameters:
            model_kwargs["output_dim"] = num_classes
        elif "out_dim" in sig.parameters:
            model_kwargs["out_dim"] = num_classes
        elif "num_classes" in sig.parameters:
            model_kwargs["num_classes"] = num_classes
            
        model = model_class(**model_kwargs)
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        from utils.protocols import EarlyStopping
        early_stopper = EarlyStopping(patience=20, verbose=False)

        # Bucle de entrenamiento con reporting de pérdida de validación para pruning
        for epoch in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item() * len(targets)
            val_loss /= len(X_val)
            
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            early_stopper(val_loss, model, epoch)
            if early_stopper.early_stop:
                model.load_state_dict(early_stopper.best_weights)
                val_loss = early_stopper.best_loss
                break
                
        return val_loss

    elif objective_type == "unlearning_loss":
        # ──────────────────────────────────────────────
        # Optimización para desaprendizaje (Unlearning)
        # ──────────────────────────────────────────────
        # Cargar el hidden_dim óptimo del modelo base desde models/best_hp.json
        best_base_hp = load_best_hp(MODELS_PATH / "best_hp.json")
        hidden_dim = best_base_hp.get("hidden_dim", 64)
        base_batch_size = best_base_hp.get("batch_size", 16)
        
        # Cargar datos de retain y forget
        data_path = DATASETS_PATH / f"{dataset}_splits_seed_{seed}.npz"
        if not data_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo de datos {data_path}. Ejecuta 2_split_dataset.py primero.")
            
        data = np.load(data_path)
        X_retain, y_retain = data["X_retain"], data["y_retain"]
        X_forget, y_forget = data["X_forget"], data["y_forget"]
        
        X_retain_t = torch.tensor(X_retain, dtype=torch.float32)
        y_retain_t = torch.tensor(y_retain, dtype=torch.long)
        X_forget_t = torch.tensor(X_forget, dtype=torch.float32)
        y_forget_t = torch.tensor(y_forget, dtype=torch.long)
        
        batch_size = suggested_hp.get("batch_size", 128) if protocol == "rurk" else base_batch_size
        # Entrenamos fine-tuning usando el batch size especificado
        train_loader = DataLoader(TensorDataset(X_retain_t, y_retain_t), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_retain_t, y_retain_t), batch_size=batch_size, shuffle=False)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_class = load_class(model_arch)
        
        # Cargar pesos del modelo base y naive pre-entrenados para la semilla de calibración
        base_weights_path = WEIGHTS_DIR / f"base_model_seed_{seed}.pth"
        naive_weights_path = WEIGHTS_DIR / f"naive_model_seed_{seed}.pth"
        
        if not base_weights_path.exists() or not naive_weights_path.exists():
            raise FileNotFoundError(
                f"Faltan los pesos de base o naive para la semilla {seed}. "
                "Por favor, entrena los modelos base y naive primero."
            )
            
        # Instanciar modelos y cargar state_dict
        num_features = X_retain.shape[1] if len(X_retain.shape) == 2 else int(np.prod(X_retain.shape[1:]))
        X_val = data["X_val"] if "X_val" in data else X_retain
        y_val = data["y_val"] if "y_val" in data else y_retain
        num_classes = int(np.max(np.concatenate([y_retain, y_forget, y_val])) + 1)
        
        sig = inspect.signature(model_class.__init__)
        
        def instantiate_model():
            model_kwargs = {}
            if "input_dim" in sig.parameters:
                model_kwargs["input_dim"] = num_features
            elif "in_dim" in sig.parameters:
                model_kwargs["in_dim"] = num_features
            elif "in_features" in sig.parameters:
                model_kwargs["in_features"] = num_features
            elif "in_channels" in sig.parameters:
                model_kwargs["in_channels"] = X_retain.shape[1]
                
            if "hidden_dim" in sig.parameters:
                model_kwargs["hidden_dim"] = hidden_dim
                
            if "output_dim" in sig.parameters:
                model_kwargs["output_dim"] = num_classes
            elif "out_dim" in sig.parameters:
                model_kwargs["out_dim"] = num_classes
            elif "num_classes" in sig.parameters:
                model_kwargs["num_classes"] = num_classes
            return model_class(**model_kwargs).to(device)
            
        naive_model = instantiate_model()
        base_model = instantiate_model()
            
        naive_model.load_state_dict(torch.load(naive_weights_path, map_location=device))
        base_model.load_state_dict(torch.load(base_weights_path, map_location=device))
        
        # Calcular precisiones absolutas del modelo naive de referencia
        naive_model.eval()
        with torch.no_grad():
            naive_retain_outputs = naive_model(X_retain_t.to(device))
            naive_retain_acc = (naive_retain_outputs.argmax(dim=1) == y_retain_t.to(device)).float().mean().item()
            
            naive_forget_outputs = naive_model(X_forget_t.to(device))
            naive_forget_acc = (naive_forget_outputs.argmax(dim=1) == y_forget_t.to(device)).float().mean().item()
            
        # Parámetros sugeridos
        lr = suggested_hp.get("lr", 1e-3)
        epochs = suggested_hp.get("epochs", 20)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(base_model.parameters(), lr=lr)
        
        # Ejecutar protocolo de olvido de forma silenciosa
        protocol_fn = get_protocol(protocol)
        protocol_sig = inspect.signature(protocol_fn)
        kwargs = {}
        if "hp" in protocol_sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in protocol_sig.parameters.values()):
            kwargs["hp"] = suggested_hp
            
        logs_dict = {}
        if protocol == "cfgk":
            kwargs["logs_dict"] = logs_dict
        elif protocol == "rurk":
            kwargs["forget_loader"] = DataLoader(TensorDataset(X_forget_t, y_forget_t), batch_size=batch_size, shuffle=True)
            
        unlearned_model = protocol_fn(
            model=base_model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            epochs=epochs,
            device=device,
            verbose=False,
            **kwargs
        )
        
        if protocol == "cfgk" and logs_dict:
            for k, v in logs_dict.items():
                trial.set_user_attr(k, v)
        
        # Evaluar el modelo desentrenado
        unlearned_model.eval()
        with torch.no_grad():
            unlearned_retain_outputs = unlearned_model(X_retain_t.to(device))
            unlearned_retain_acc = (unlearned_retain_outputs.argmax(dim=1) == y_retain_t.to(device)).float().mean().item()
            
            unlearned_forget_outputs = unlearned_model(X_forget_t.to(device))
            unlearned_forget_acc = (unlearned_forget_outputs.argmax(dim=1) == y_forget_t.to(device)).float().mean().item()
            
        # Loss: penaliza pérdida de retención y exceso de retención sobre el forget set
        retain_diff = naive_retain_acc - unlearned_retain_acc
        forget_excess = unlearned_forget_acc - naive_forget_acc
        
        loss = (retain_diff ** 2) + max(0.0, forget_excess) ** 2
        return loss

    else:
        raise ValueError(f"Tipo de objetivo '{objective_type}' no soportado.")


def run_search(protocol: str, n_trials: int, seed: int, model_arch: str, dataset: str = "spiral"):
    """
    Crea el estudio Optuna, ejecuta la optimización y guarda los resultados en JSON.
    """
    if protocol not in HP_SPACES:
        raise ValueError(f"Protocolo '{protocol}' no soportado.")
        
    hp_config = HP_SPACES[protocol]
    output_path = Path(hp_config["output_path"])
    
    print("\n" + "=" * 80)
    print(f" INICIANDO BÚSQUEDA DE HIPERPARÁMETROS CON OPTUNA ")
    print(f" Protocolo: {protocol} | Dirección: Minimizar | Trials: {n_trials} | Semilla: {seed} | Dataset: {dataset}")
    print("=" * 80 + "\n")
    
    # Crear un estudio Optuna
    study = optuna.create_study(
        direction="minimize",
        study_name=f"{protocol}_hp_search",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20),
    )
    
    study.optimize(
        lambda trial: objective(trial, protocol, seed, model_arch, dataset),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    best = study.best_trial
    print("\n" + "=" * 80)
    print("  MEJORES HIPERPARÁMETROS ENCONTRADOS")
    print("=" * 80)
    print(f"  Valor Objetivo Mínimo: {best.value:.6f}")
    for key, value in best.params.items():
        print(f"  {key:20s}: {value}")
    print("=" * 80 + "\n")
    
    # Añadir valores implícitos de configuración si los hay
    # (Por ejemplo, en CFK epochs es fijo a 20 y usa el hidden_dim de la base)
    suggested_params = dict(best.params)
    if protocol in ["cfk", "euk", "cfgk", "rurk"]:
        if protocol == "cfgk":
            suggested_params["epochs"] = 10
        elif protocol == "rurk":
            suggested_params["epochs"] = 2
        else:
            suggested_params["epochs"] = 20
        best_base_hp = load_best_hp(MODELS_PATH / "best_hp.json")
        suggested_params["hidden_dim"] = best_base_hp.get("hidden_dim", 64)
        
    # Guardar a JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(suggested_params, f, indent=2, ensure_ascii=False)
    print(f"Hiperparámetros guardados exitosamente en: {output_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Búsqueda generalizada de hiperparámetros con Optuna.")
    parser.add_argument(
        "--protocol",
        type=str,
        default="standard",
        choices=["standard", "cfk", "euk", "cfgk", "rurk"],
        help="Nombre del protocolo a buscar en HP_SPACES (default: standard)."
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=30,
        help="Número de trials de optimización a ejecutar (default: 30)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Semilla del split del dataset para calibración (default: 0)."
    )
    parser.add_argument(
        "--model_arch",
        type=str,
        default="models.base_nn.BaseMLP",
        help="Importación de la clase del modelo (default: models.base_nn.BaseMLP)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="spiral",
        help="Nombre o prefijo del dataset (default: spiral)."
    )
    args = parser.parse_args()
    
    run_search(
        protocol=args.protocol,
        n_trials=args.n_trials,
        seed=args.seed,
        model_arch=args.model_arch,
        dataset=args.dataset
    )
