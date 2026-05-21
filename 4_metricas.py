"""
4_metricas.py
───────────────────────────────────────────────
Paso 4 del pipeline: Evaluación cuantitativa de desentrenamiento (Unlearning).

Calcula y reporta las precisiones absolutas sobre los conjuntos de datos, así como:
- RR (Retain Ratio) = Acc(Modelo, Retain) / Acc(Naive, Retain)
- RF (Forget Ratio) = Acc(Modelo, Forget) / Acc(Naive, Forget)
- RT (Test Ratio) = Acc(Modelo, Test) / Acc(Naive, Test)
- TR (Time Ratio) = Tiempo(Modelo) / Tiempo(Naive)

Los ratios se calculan para todos los modelos (Base, Naive y Unlearned).
Para el modelo Naive de referencia, todos los ratios son lógicamente 1.0.

Los resultados se guardan en formato JSON en results/{unlearned_name}_metrics.json
"""

import argparse
import json
import inspect
import importlib
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from utils.config import DATASETS_PATH


def load_class(class_path: str):
    """Carga dinámicamente una clase a partir de su string de importación."""
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except Exception as e:
        raise ImportError(f"No se pudo cargar la clase '{class_path}': {e}")


def load_best_hp(hp_file_path: Path) -> dict:
    """Carga los mejores hiperparámetros desde un archivo JSON si existe."""
    if not hp_file_path.exists():
        return {}
    try:
        with open(hp_file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def evaluate_model(model: nn.Module, X: np.ndarray, y: np.ndarray, device: torch.device, batch_size: int = 64) -> float:
    """
    Evalúa la precisión (Accuracy) de un modelo sobre los datos entregados.
    """
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    
    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    return correct / total if total > 0 else 0.0


def calculate_metrics(
    unlearned_name: str = "cfk",
    base_name: str = "base",
    naive_name: str = "naive",
    model_arch: str = "models.base_nn.BaseMLP",
    seeds: list[int] = [0, 1, 2],
    hp: dict = None,
    output_dir: str = "results",
    weights_dir: str = "models/weights",
    dataset: str = "spiral"
) -> dict:
    """
    Calcula y reporta la precisión absoluta y los ratios relacionales (RR, RF, RT, TR) por semilla.
    """
    if hp is None:
        hp = {}
    hidden_dim = hp.get("hidden_dim", 16)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_class = load_class(model_arch)
    
    results = {
        "unlearned_name": unlearned_name,
        "seeds": seeds,
        "per_seed": {},
        "aggregated": {}
    }
    
    # Listas para almacenar métricas para calcular agregación posterior
    metrics_list = {
        "base_retain": [], "base_forget": [], "base_test": [],
        "base_RR": [], "base_RF": [], "base_RT": [],
        "base_epochs": [], "base_time": [], "base_TR": [],
        
        "naive_retain": [], "naive_forget": [], "naive_test": [],
        "naive_RR": [], "naive_RF": [], "naive_RT": [],
        "naive_epochs": [], "naive_time": [], "naive_TR": [],
        
        "unlearned_retain": [], "unlearned_forget": [], "unlearned_test": [],
        "unlearned_RR": [], "unlearned_RF": [], "unlearned_RT": [],
        "unlearned_epochs": [], "unlearned_time": [], "unlearned_TR": []
    }
    
    print("\n" + "="*80)
    print(f" EVALUACIÓN DE MÉTRICAS DE OLVIDO PARA EL MÉTODO: {unlearned_name.upper()} ")
    print("="*80)
    
    for seed in seeds:
        # Cargar datos
        data_path = DATASETS_PATH / f"{dataset}_splits_seed_{seed}.npz"
        if not data_path.exists():
            raise FileNotFoundError(f"No se encontró el split para semilla {seed} en {data_path}")
        
        data = np.load(data_path)
        X_retain, y_retain = data["X_retain"], data["y_retain"]
        X_forget, y_forget = data["X_forget"], data["y_forget"]
        X_test, y_test = data["X_test"], data["y_test"]
        
        # Determinar dimensiones dinámicamente
        num_features = X_retain.shape[1] if len(X_retain.shape) == 2 else int(np.prod(X_retain.shape[1:]))
        all_y = np.concatenate([y_retain, y_forget, y_test])
        num_classes = int(np.max(all_y) + 1)
        
        # Cargar Modelos y metadatos
        models = {}
        metadata_vals = {}
        for name, key in [(base_name, "base"), (naive_name, "naive"), (unlearned_name, "unlearned")]:
            # Instanciar arquitectura
            sig = inspect.signature(model_class.__init__)
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
                
            if "pretrained" in sig.parameters and "pretrained" in hp:
                model_kwargs["pretrained"] = hp["pretrained"]
                
            m = model_class(**model_kwargs)
            
            m = m.to(device)
            
            weights_path = Path(weights_dir) / f"{name}_model_seed_{seed}.pth"
            if not weights_path.exists():
                raise FileNotFoundError(f"Pesos no encontrados para el modelo '{name}' en semilla {seed}: {weights_path}")
                
            m.load_state_dict(torch.load(weights_path, map_location=device))
            models[key] = m
            
            # Cargar metadatos
            meta_path = Path(weights_dir) / f"{name}_model_seed_{seed}_meta.json"
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    metadata_vals[f"{key}_epochs"] = meta.get("epochs", 0)
                    metadata_vals[f"{key}_time"] = meta.get("time_elapsed", 0.0)
                except Exception:
                    metadata_vals[f"{key}_epochs"] = 0
                    metadata_vals[f"{key}_time"] = 0.0
            else:
                metadata_vals[f"{key}_epochs"] = 0
                metadata_vals[f"{key}_time"] = 0.0
            
        # Evaluar precisiones absolutas
        accs = {}
        for key in ["base", "naive", "unlearned"]:
            accs[f"{key}_retain"] = evaluate_model(models[key], X_retain, y_retain, device)
            accs[f"{key}_forget"] = evaluate_model(models[key], X_forget, y_forget, device)
            accs[f"{key}_test"] = evaluate_model(models[key], X_test, y_test, device)
            
        # Calcular Ratios respecto a Naive (RR, RF, RT)
        ratios = {}
        for key in ["base", "naive", "unlearned"]:
            # Retain Ratio (RR)
            if accs["naive_retain"] > 0:
                rr = accs[f"{key}_retain"] / accs["naive_retain"]
            else:
                rr = 1.0 if accs[f"{key}_retain"] == 0 else float("inf")
                
            # Forget Ratio (RF)
            if accs["naive_forget"] > 0:
                rf = accs[f"{key}_forget"] / accs["naive_forget"]
            else:
                rf = 1.0 if accs[f"{key}_forget"] == 0 else float("inf")
                
            # Test Ratio (RT)
            if accs["naive_test"] > 0:
                rt = accs[f"{key}_test"] / accs["naive_test"]
            else:
                rt = 1.0 if accs[f"{key}_test"] == 0 else float("inf")
                
            ratios[f"{key}_RR"] = rr
            ratios[f"{key}_RF"] = rf
            ratios[f"{key}_RT"] = rt
            
        # Calcular Time Ratios (TR) respecto a Naive
        time_ratios = {}
        naive_time = metadata_vals["naive_time"]
        for key in ["base", "naive", "unlearned"]:
            model_time = metadata_vals[f"{key}_time"]
            if naive_time > 0:
                tr = model_time / naive_time
            else:
                tr = 1.0 if model_time == 0 else float("inf")
            time_ratios[f"{key}_TR"] = tr
            
        # Guardar en per_seed
        results["per_seed"][str(seed)] = {
            "base": {
                "retain": accs["base_retain"],
                "forget": accs["base_forget"],
                "test": accs["base_test"],
                "epochs": metadata_vals["base_epochs"],
                "time": metadata_vals["base_time"],
                "RR": ratios["base_RR"],
                "RF": ratios["base_RF"],
                "RT": ratios["base_RT"],
                "TR": time_ratios["base_TR"]
            },
            "naive": {
                "retain": accs["naive_retain"],
                "forget": accs["naive_forget"],
                "test": accs["naive_test"],
                "epochs": metadata_vals["naive_epochs"],
                "time": metadata_vals["naive_time"],
                "RR": ratios["naive_RR"],
                "RF": ratios["naive_RF"],
                "RT": ratios["naive_RT"],
                "TR": time_ratios["naive_TR"]
            },
            "unlearned": {
                "retain": accs["unlearned_retain"],
                "forget": accs["unlearned_forget"],
                "test": accs["unlearned_test"],
                "epochs": metadata_vals["unlearned_epochs"],
                "time": metadata_vals["unlearned_time"],
                "RR": ratios["unlearned_RR"],
                "RF": ratios["unlearned_RF"],
                "RT": ratios["unlearned_RT"],
                "TR": time_ratios["unlearned_TR"]
            }
        }
        
        # Almacenar en listas agregadas
        for k_met in metrics_list.keys():
            key, met = k_met.split("_", 1)
            if met in ["retain", "forget", "test"]:
                metrics_list[k_met].append(accs[f"{key}_{met}"])
            elif met in ["epochs", "time"]:
                metrics_list[k_met].append(metadata_vals[k_met])
            elif met == "TR":
                metrics_list[k_met].append(time_ratios[k_met])
            else:
                metrics_list[k_met].append(ratios[k_met])
                
        print(f"\n[Semilla {seed}]")
        print(f"  Base Model  | Retain Acc: {accs['base_retain']*100:.2f}% | Forget Acc: {accs['base_forget']*100:.2f}% | Test Acc: {accs['base_test']*100:.2f}%")
        print(f"  --> Ratios  | RR: {ratios['base_RR']:.4f} | RF: {ratios['base_RF']:.4f} | RT: {ratios['base_RT']:.4f}")
        print(f"  --> Tiempo  | Epochs: {metadata_vals['base_epochs']} | Time: {metadata_vals['base_time']:.4f}s | TR: {time_ratios['base_TR']:.4f}")
        
        print(f"  Naive Model | Retain Acc: {accs['naive_retain']*100:.2f}% | Forget Acc: {accs['naive_forget']*100:.2f}% | Test Acc: {accs['naive_test']*100:.2f}%")
        print(f"  --> Ratios  | RR: {ratios['naive_RR']:.4f} | RF: {ratios['naive_RF']:.4f} | RT: {ratios['naive_RT']:.4f}")
        print(f"  --> Tiempo  | Epochs: {metadata_vals['naive_epochs']} | Time: {metadata_vals['naive_time']:.4f}s | TR: {time_ratios['naive_TR']:.4f}")
        
        print(f"  Unlearned   | Retain Acc: {accs['unlearned_retain']*100:.2f}% | Forget Acc: {accs['unlearned_forget']*100:.2f}% | Test Acc: {accs['unlearned_test']*100:.2f}%")
        print(f"  --> Ratios  | RR: {ratios['unlearned_RR']:.4f} | RF: {ratios['unlearned_RF']:.4f} | RT: {ratios['unlearned_RT']:.4f}")
        print(f"  --> Tiempo  | Epochs: {metadata_vals['unlearned_epochs']} | Time: {metadata_vals['unlearned_time']:.4f}s | TR: {time_ratios['unlearned_TR']:.4f}")

    # Agregación (Media ± Desviación Estándar)
    for k_met, values in metrics_list.items():
        results["aggregated"][k_met] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values))
        }
        
    print("\n" + "="*80)
    print(" RESUMEN AGREGADO (Media ± Desv. Estándar) ")
    print("="*80)
    
    def fmt_aggr(key):
        mean = results["aggregated"][key]["mean"] * 100
        std = results["aggregated"][key]["std"] * 100
        return f"{mean:.2f}% ± {std:.2f}%"
        
    def fmt_ratio(key):
        mean = results["aggregated"][key]["mean"]
        std = results["aggregated"][key]["std"]
        return f"{mean:.4f} ± {std:.4f}"
        
    def fmt_time(key):
        mean = results["aggregated"][key]["mean"]
        std = results["aggregated"][key]["std"]
        return f"{mean:.4f}s ± {std:.4f}s"
        
    def fmt_epochs(key):
        mean = results["aggregated"][key]["mean"]
        return f"{mean:.1f}"
        
    print(f"Base Model  | Retain: {fmt_aggr('base_retain')} | Forget: {fmt_aggr('base_forget')} | Test: {fmt_aggr('base_test')}")
    print(f"            | RR: {fmt_ratio('base_RR')} | RF: {fmt_ratio('base_RF')} | RT: {fmt_ratio('base_RT')}")
    print(f"            | Epochs: {fmt_epochs('base_epochs')} | Time: {fmt_time('base_time')} | TR: {fmt_ratio('base_TR')}")
    print("-" * 80)
    print(f"Naive Model | Retain: {fmt_aggr('naive_retain')} | Forget: {fmt_aggr('naive_forget')} | Test: {fmt_aggr('naive_test')}")
    print(f"            | RR: {fmt_ratio('naive_RR')} | RF: {fmt_ratio('naive_RF')} | RT: {fmt_ratio('naive_RT')}")
    print(f"            | Epochs: {fmt_epochs('naive_epochs')} | Time: {fmt_time('naive_time')} | TR: {fmt_ratio('naive_TR')}")
    print("-" * 80)
    print(f"Unlearned   | Retain: {fmt_aggr('unlearned_retain')} | Forget: {fmt_aggr('unlearned_forget')} | Test: {fmt_aggr('unlearned_test')}")
    print(f"            | RR: {fmt_ratio('unlearned_RR')} | RF: {fmt_ratio('unlearned_RF')} | RT: {fmt_ratio('unlearned_RT')}")
    print(f"            | Epochs: {fmt_epochs('unlearned_epochs')} | Time: {fmt_time('unlearned_time')} | TR: {fmt_ratio('unlearned_TR')}")
    print("="*80)
    
    # Guardar en archivo JSON
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    out_file = out_dir_path / f"{unlearned_name}_metrics.json"
    
    # Manejar posibles valores infinitos o NaN para guardado JSON
    def serialize_val(v):
        if isinstance(v, float):
            if np.isinf(v) or np.isnan(v):
                return str(v)
        return v
        
    cleaned_results = json.loads(
        json.dumps(results, default=serialize_val)
    )
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_results, f, indent=4, ensure_ascii=False)
    print(f"Métricas guardadas exitosamente en: {out_file}\n")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluación de métricas de unlearning.")
    parser.add_argument("--unlearned_name", type=str, default="cfk",
                        help="Nombre del modelo desentrenado a evaluar (ej: cfk)")
    parser.add_argument("--base_name", type=str, default="base",
                        help="Prefijo del modelo base completo (ej: base)")
    parser.add_argument("--naive_name", type=str, default="naive",
                        help="Prefijo del modelo naive (ej: naive)")
    parser.add_argument("--model_arch", type=str, default="models.base_nn.BaseMLP",
                        help="Import path de la arquitectura a instanciar")
    parser.add_argument("--seeds", type=str, default="0,1,2",
                        help="Semillas a evaluar separadas por comas (ej: 0,1,2)")
    parser.add_argument("--hp_file", type=str, default="models/best_hp.json",
                        help="Ruta al JSON de hiperparámetros")
    parser.add_argument("--hidden_dim", type=int,
                        help="Dimensión oculta del modelo (sobrascribe hp_file)")
    parser.add_argument("--weights_dir", type=str, default="models/weights",
                        help="Directorio donde se encuentran los pesos y metadatos de los modelos")
    parser.add_argument("--dataset", type=str, default="spiral",
                        help="Nombre o prefijo del dataset (default: spiral)")
                        
    args = parser.parse_args()
    
    # Cargar mejores hiperparámetros
    best_hp = load_best_hp(Path(args.hp_file))
    
    hp = {
        "hidden_dim": args.hidden_dim if args.hidden_dim is not None else best_hp.get("hidden_dim", 16)
    }
    
    seeds_list = [int(s.strip()) for s in args.seeds.split(",")]
    
    calculate_metrics(
        unlearned_name=args.unlearned_name,
        base_name=args.base_name,
        naive_name=args.naive_name,
        model_arch=args.model_arch,
        seeds=seeds_list,
        hp=hp,
        output_dir=args.output_dir,
        weights_dir=args.weights_dir,
        dataset=args.dataset
    )
