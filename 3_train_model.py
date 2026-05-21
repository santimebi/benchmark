"""
3_train_model.py
───────────────────────────────────────────────
Paso 3 del pipeline: Entrenamiento genérico y parametrizado de modelos.

Entrena un modelo especificando la arquitectura, el protocolo de entrenamiento,
los splits del dataset a incluir y los hiperparámetros del modelo.

Por defecto, entrena el modelo "base" (Retain + Forget), pero puede utilizarse
para entrenar el modelo "naive" (solo Retain) o aplicar protocolos de unlearning.

Uso:
    python 3_train_model.py --model_name base --train_splits retain,forget
    python 3_train_model.py --model_name naive --train_splits retain
"""

import argparse
import json
import inspect
import importlib
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from utils.config import DATASETS_PATH

WEIGHTS_DIR = Path("models/weights")


def load_class(class_path: str):
    """
    Carga dinámicamente una clase a partir de su string de importación.

    Args:
        class_path: Ruta de importación de la clase (ej. 'models.base_nn.BaseMLP').

    Returns:
        type: La clase cargada.
    """
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except Exception as e:
        raise ImportError(f"No se pudo cargar la clase '{class_path}': {e}")


def load_dataset_splits(data_path: Path, splits: list[str]) -> tuple:
    """
    Carga y concatena los splits indicados desde un archivo .npz de datasets.

    Args:
        data_path: Ruta al archivo .npz de splits.
        splits: Lista de splits a combinar para el entrenamiento (ej. ['retain', 'forget']).

    Returns:
        tuple: (X_train, y_train, X_val, y_val)
    """
    if not data_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset en {data_path}. Ejecuta 2_split_dataset.py primero.")
    
    data = np.load(data_path)
    X_list = []
    y_list = []
    
    for split in splits:
        split_clean = split.strip()
        X_key = f"X_{split_clean}"
        y_key = f"y_{split_clean}"
        
        if X_key not in data or y_key not in data:
            raise KeyError(f"El split '{split_clean}' no existe en {data_path.name}")
            
        X_list.append(data[X_key])
        y_list.append(data[y_key])
        
    X_train = np.vstack(X_list)
    y_train = np.concatenate(y_list)
    
    X_val = data['X_val']
    y_val = data['y_val']
    
    return X_train, y_train, X_val, y_val


def load_best_hp(hp_file_path: Path) -> dict:
    """
    Carga los mejores hiperparámetros desde un archivo JSON si existe.
    """
    if not hp_file_path.exists():
        print(f"[WARN] No se encontró {hp_file_path}. Usando hiperparámetros por defecto.")
        return {}
    with open(hp_file_path, "r", encoding="utf-8") as f:
        hp = json.load(f)
    print(f"Hiperparámetros cargados desde {hp_file_path}: {hp}")
    return hp


def train_model(
    model_arch: str = "models.base_nn.BaseMLP",
    protocol: str = "standard",
    train_splits: list[str] = ["retain", "forget"],
    seed: int = 0,
    model_name: str = "base",
    hp: dict = None,
    verbose: bool = True,
    pretrained_weights: str = None,
    dataset: str = "spiral"
) -> nn.Module:
    """
    Entrena un modelo configurable sobre los splits indicados con un protocolo dado.

    Args:
        model_arch: Ruta de importación de la arquitectura del modelo.
        protocol: Identificador del protocolo de entrenamiento/olvido.
        train_splits: Lista de splits a combinar en el conjunto de entrenamiento.
        seed: Semilla del split del dataset a cargar.
        model_name: Prefijo para guardar los pesos del modelo.
        hp: Diccionario de hiperparámetros opcionales.
        verbose: Si es True, imprime progreso de entrenamiento.
        pretrained_weights: Ruta a los pesos pre-entrenados del modelo (puede usar {seed}).
        dataset: Nombre o prefijo del dataset a cargar.

    Returns:
        nn.Module: Modelo entrenado.
    """
    if hp is None:
        hp = {}
        
    epochs = hp.get("epochs", 150)
    batch_size = hp.get("batch_size", 32)
    lr = hp.get("lr", 1e-3)
    hidden_dim = hp.get("hidden_dim", 16)
    
    # 1. Cargar datos del split específico
    data_path = DATASETS_PATH / f"{dataset}_splits_seed_{seed}.npz"
    X_train, y_train, X_val, y_val = load_dataset_splits(data_path, train_splits)
    
    # Convertir a tensores
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    
    # Crear DataLoaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val_t, y_val_t)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. Instanciar el modelo
    model_class = load_class(model_arch)
    
    # Determinar input_dim y output_dim dinámicamente de los datos cargados
    num_features = X_train.shape[1] if len(X_train.shape) == 2 else int(np.prod(X_train.shape[1:]))
    all_y = np.concatenate([y_train, y_val])
    num_classes = int(np.max(all_y) + 1)

    sig = inspect.signature(model_class.__init__)
    model_kwargs = {}
    
    # Mapeo de nombres comunes para la entrada
    if "input_dim" in sig.parameters:
        model_kwargs["input_dim"] = num_features
    elif "in_dim" in sig.parameters:
        model_kwargs["in_dim"] = num_features
    elif "in_features" in sig.parameters:
        model_kwargs["in_features"] = num_features
    elif "in_channels" in sig.parameters:
        model_kwargs["in_channels"] = X_train.shape[1]
        
    # Mapeo de nombres comunes para la dimensión oculta
    if "hidden_dim" in sig.parameters:
        model_kwargs["hidden_dim"] = hidden_dim
        
    # Mapeo de nombres comunes para la salida (clases)
    if "output_dim" in sig.parameters:
        model_kwargs["output_dim"] = num_classes
    elif "out_dim" in sig.parameters:
        model_kwargs["out_dim"] = num_classes
    elif "num_classes" in sig.parameters:
        model_kwargs["num_classes"] = num_classes

    # Para otros kwargs que el constructor pueda aceptar (ej. pretrained)
    if "pretrained" in sig.parameters and "pretrained" in hp:
        model_kwargs["pretrained"] = hp["pretrained"]
        
    model = model_class(**model_kwargs)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Cargar pesos pre-entrenados si se especifican
    if pretrained_weights:
        formatted_weights_path = Path(pretrained_weights.format(seed=seed))
        if formatted_weights_path.exists():
            model.load_state_dict(torch.load(formatted_weights_path, map_location=device))
            if verbose:
                print(f"Pesos pre-entrenados cargados desde: {formatted_weights_path}")
        else:
            raise FileNotFoundError(f"No se encontró el archivo de pesos pre-entrenados en: {formatted_weights_path}")
            
    # 3. Configurar optimizador y función de pérdida
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    if verbose:
        print(f"--- Iniciando entrenamiento del modelo '{model_name}' (Seed: {seed}) ---")
        print(f"  Arquitectura: {model_arch} | Protocolo: {protocol}")
        print(f"  Splits de entrenamiento: {train_splits} (Muestras: {len(X_train)})")
        print(f"  Hiperparámetros: epochs={epochs}, batch_size={batch_size}, lr={lr}, hidden_dim={hidden_dim}")
        
    # 4. Obtener protocolo y entrenar
    from utils.protocols import get_protocol
    protocol_fn = get_protocol(protocol)
    
    # Comprobar si la función acepta hp o kwargs
    protocol_sig = inspect.signature(protocol_fn)
    kwargs = {}
    if "hp" in protocol_sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in protocol_sig.parameters.values()):
        kwargs["hp"] = hp
        
    start_time = time.perf_counter()
    model = protocol_fn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs,
        device=device,
        verbose=verbose,
        **kwargs
    )
    elapsed_time = time.perf_counter() - start_time
    
    # 5. Guardar el modelo y sus metadatos
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = WEIGHTS_DIR / f"{model_name}_model_seed_{seed}.pth"
    torch.save(model.state_dict(), model_path)
    
    meta_path = WEIGHTS_DIR / f"{model_name}_model_seed_{seed}_meta.json"
    meta_data = {
        "epochs": epochs,
        "time_elapsed": elapsed_time
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=2, ensure_ascii=False)
    
    if verbose:
        print(f"Modelo guardado exitosamente en: {model_path}")
        print(f"Metadatos guardados exitosamente en: {meta_path} (Tiempo: {elapsed_time:.4f}s)")
        
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento genérico del benchmark.")
    parser.add_argument("--model_arch", type=str, default="models.base_nn.BaseMLP", 
                        help="Import path de la clase del modelo (ej. models.base_nn.BaseMLP)")
    parser.add_argument("--protocol", type=str, default="standard", 
                        help="Protocolo de entrenamiento o unlearning a ejecutar (ej. standard)")
    parser.add_argument("--train_splits", type=str, default="retain,forget", 
                        help="Splits separados por comas a incluir en el entrenamiento (ej. retain,forget)")
    parser.add_argument("--model_name", type=str, default="base", 
                        help="Prefijo para nombrar el archivo de pesos resultante")
    parser.add_argument("--hp_file", type=str, default="models/best_hp.json", 
                        help="Ruta al archivo JSON de mejores hiperparámetros")
    parser.add_argument("--epochs", type=int, help="Número de épocas de entrenamiento (sobrescribe hp_file)")
    parser.add_argument("--batch_size", type=int, help="Tamaño de batch (sobrescribe hp_file)")
    parser.add_argument("--lr", type=float, help="Learning rate (sobrescribe hp_file)")
    parser.add_argument("--hidden_dim", type=int, help="Dimensión oculta del modelo (sobrescribe hp_file)")
    parser.add_argument("--seeds", type=str, default="0,1,2", 
                        help="Semillas para las cuales entrenar el modelo, separadas por comas (ej. 0,1,2)")
    parser.add_argument("--pretrained_weights", type=str, default=None,
                        help="Ruta a los pesos pre-entrenados del modelo (puede usar {seed} para formatear dinámicamente)")
    parser.add_argument("--k", type=int, default=None,
                        help="Número de últimas capas a descongelar para el protocolo cfk")
    parser.add_argument("--dataset", type=str, default="spiral",
                        help="Nombre o prefijo del dataset a cargar (ej. spiral, cifar10)")
    parser.add_argument("--no_verbose", dest="verbose", action="store_false", 
                        help="Desactiva los mensajes y logs de progreso")
    parser.set_defaults(verbose=True)
    
    args = parser.parse_args()
    
    # Cargar mejores hiperparámetros
    best_hp = load_best_hp(Path(args.hp_file))
    
    # Combinar valores del fichero con los argumentos explícitos de CLI
    hp = {
        "epochs": args.epochs if args.epochs is not None else best_hp.get("epochs", 150),
        "batch_size": args.batch_size if args.batch_size is not None else best_hp.get("batch_size", 32),
        "lr": args.lr if args.lr is not None else best_hp.get("lr", 1e-3),
        "hidden_dim": args.hidden_dim if args.hidden_dim is not None else best_hp.get("hidden_dim", 16),
        "k": args.k if args.k is not None else best_hp.get("k", 1),
    }
    
    # Procesar listas
    splits_list = [s.strip() for s in args.train_splits.split(",")]
    seeds_list = [int(s.strip()) for s in args.seeds.split(",")]
    
    # Entrenar para cada semilla
    for seed in seeds_list:
        train_model(
            model_arch=args.model_arch,
            protocol=args.protocol,
            train_splits=splits_list,
            seed=seed,
            model_name=args.model_name,
            hp=hp,
            verbose=args.verbose,
            pretrained_weights=args.pretrained_weights,
            dataset=args.dataset
        )
        if args.verbose:
            print("-" * 50)
