"""
4_metricas.py
───────────────────────────────────────────────
Paso 4 del pipeline: Evaluación cuantitativa de desentrenamiento (Unlearning).

Calcula y reporta las precisiones absolutas sobre los conjuntos de datos, así como:
- RA (Retain Accuracy) = Acc(Unlearned, Retain) / Acc(Naive, Retain)
- FA (Forget Accuracy) = Acc(Unlearned, Forget) / Acc(Naive, Forget)

Los resultados se guardan en format JSON en results/{unlearned_name}_metrics.json
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
    output_dir: str = "results"
) -> dict:
    """
    Calcula y reporta la precisión absoluta y las métricas relativas (RA, FA) por semilla.
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
        "naive_retain": [], "naive_forget": [], "naive_test": [],
        "unlearned_retain": [], "unlearned_forget": [], "unlearned_test": [],
        "RA": [], "FA": []
    }
    
    print("\n" + "="*80)
    print(f" EVALUACIÓN DE MÉTRICAS DE OLVIDO PARA EL MÉTODO: {unlearned_name.upper()} ")
    print("="*80)
    
    for seed in seeds:
        # Cargar datos
        data_path = DATASETS_PATH / f"spiral_splits_seed_{seed}.npz"
        if not data_path.exists():
            raise FileNotFoundError(f"No se encontró el split para semilla {seed} en {data_path}")
        
        data = np.load(data_path)
        X_retain, y_retain = data["X_retain"], data["y_retain"]
        X_forget, y_forget = data["X_forget"], data["y_forget"]
        X_test, y_test = data["X_test"], data["y_test"]
        
        # Cargar Modelos
        models = {}
        for name, key in [(base_name, "base"), (naive_name, "naive"), (unlearned_name, "unlearned")]:
            # Instanciar arquitectura
            sig = inspect.signature(model_class.__init__)
            if "hidden_dim" in sig.parameters:
                m = model_class(input_dim=2, hidden_dim=hidden_dim, output_dim=3)
            else:
                m = model_class(input_dim=2, output_dim=3)
            
            m = m.to(device)
            
            weights_path = Path(f"models/weights/{name}_model_seed_{seed}.pth")
            if not weights_path.exists():
                raise FileNotFoundError(f"Pesos no encontrados para el modelo '{name}' en semilla {seed}: {weights_path}")
                
            m.load_state_dict(torch.load(weights_path, map_location=device))
            models[key] = m
            
        # Evaluar
        accs = {}
        for key in ["base", "naive", "unlearned"]:
            accs[f"{key}_retain"] = evaluate_model(models[key], X_retain, y_retain, device)
            accs[f"{key}_forget"] = evaluate_model(models[key], X_forget, y_forget, device)
            accs[f"{key}_test"] = evaluate_model(models[key], X_test, y_test, device)
            
        # Métricas relativas
        # Retain Accuracy (RA)
        if accs["naive_retain"] > 0:
            ra = accs["unlearned_retain"] / accs["naive_retain"]
        else:
            ra = 1.0 if accs["unlearned_retain"] == 0 else float("inf")
            
        # Forget Accuracy (FA)
        if accs["naive_forget"] > 0:
            fa = accs["unlearned_forget"] / accs["naive_forget"]
        else:
            fa = 1.0 if accs["unlearned_forget"] == 0 else float("inf")
            
        # Guardar en per_seed
        results["per_seed"][str(seed)] = {
            "base": {"retain": accs["base_retain"], "forget": accs["base_forget"], "test": accs["base_test"]},
            "naive": {"retain": accs["naive_retain"], "forget": accs["naive_forget"], "test": accs["naive_test"]},
            "unlearned": {"retain": accs["unlearned_retain"], "forget": accs["unlearned_forget"], "test": accs["unlearned_test"]},
            "RA": ra,
            "FA": fa
        }
        
        # Almacenar en listas agregadas
        for k_met in metrics_list.keys():
            if k_met == "RA":
                metrics_list["RA"].append(ra)
            elif k_met == "FA":
                metrics_list["FA"].append(fa)
            else:
                metrics_list[k_met].append(accs[k_met])
                
        print(f"\n[Semilla {seed}]")
        print(f"  Base Model  | Retain Acc: {accs['base_retain']*100:.2f}% | Forget Acc: {accs['base_forget']*100:.2f}% | Test Acc: {accs['base_test']*100:.2f}%")
        print(f"  Naive Model | Retain Acc: {accs['naive_retain']*100:.2f}% | Forget Acc: {accs['naive_forget']*100:.2f}% | Test Acc: {accs['naive_test']*100:.2f}%")
        print(f"  Unlearned   | Retain Acc: {accs['unlearned_retain']*100:.2f}% | Forget Acc: {accs['unlearned_forget']*100:.2f}% | Test Acc: {accs['unlearned_test']*100:.2f}%")
        print(f"  --> RA (Retain Acc Relativa): {ra:.4f} (Ideal: ~1.0)")
        print(f"  --> FA (Forget Acc Relativa): {fa:.4f} (Ideal: similar al Naive, ej: ~{accs['naive_forget']/accs['naive_forget'] if accs['naive_forget'] > 0 else 1.0:.1f})")

    # Agregación
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
        
    print(f"Base Model  | Retain: {fmt_aggr('base_retain')} | Forget: {fmt_aggr('base_forget')} | Test: {fmt_aggr('base_test')}")
    print(f"Naive Model | Retain: {fmt_aggr('naive_retain')} | Forget: {fmt_aggr('naive_forget')} | Test: {fmt_aggr('naive_test')}")
    print(f"Unlearned   | Retain: {fmt_aggr('unlearned_retain')} | Forget: {fmt_aggr('unlearned_forget')} | Test: {fmt_aggr('unlearned_test')}")
    
    ra_mean = results["aggregated"]["RA"]["mean"]
    ra_std = results["aggregated"]["RA"]["std"]
    fa_mean = results["aggregated"]["FA"]["mean"]
    fa_std = results["aggregated"]["FA"]["std"]
    print(f"\n--> RA Promedio: {ra_mean:.4f} ± {ra_std:.4f}")
    print(f"--> FA Promedio: {fa_mean:.4f} ± {fa_std:.4f}")
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
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directorio de guardado para el JSON final")
                        
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
        output_dir=args.output_dir
    )
