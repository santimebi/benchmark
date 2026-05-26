"""
plot_boundaries.py
───────────────────────────────────────────────
Genera un gráfico en cuadrícula de 3x3 que muestra las fronteras de decisión y las
regiones de clasificación para los modelos Base, Naive y Unlearned (CFK o EUK)
a lo largo de las semillas 0, 1 y 2.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from utils.config import DATASETS_PATH, MODELS_PATH, RESULTS_PATH
import importlib

def load_class(class_path: str):
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except Exception as e:
        raise ImportError(f"No se pudo cargar la clase '{class_path}': {e}")

def main():
    parser = argparse.ArgumentParser(description="Graficar fronteras de decisión en un grid de 3x3.")
    parser.add_argument("--unlearned_name", type=str, default="cfk", choices=["cfk", "euk", "cfgk"],
                        help="Nombre del método de desaprendizaje (cfk o euk)")
    parser.add_argument("--model_arch", type=str, default="models.base_nn.BaseMLP",
                        help="Arquitectura del modelo")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Ruta de guardado alternativa")
    args = parser.parse_args()

    # Cargar mejores hiperparámetros
    hp_path = MODELS_PATH / "best_hp.json"
    hidden_dim = 16
    if hp_path.exists():
        try:
            with open(hp_path, "r", encoding="utf-8") as f:
                hp = json.load(f)
                hidden_dim = hp.get("hidden_dim", 16)
        except Exception:
            pass

    # Cargar clase del modelo
    model_class = load_class(args.model_arch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seeds = [0, 1, 2]
    model_types = [
        ("base", "Modelo Base (Retain + Forget)"),
        ("naive", "Modelo Naive (Reference)"),
        (args.unlearned_name, f"Modelo Unlearned ({args.unlearned_name.upper()})")
    ]

    # Paleta de colores para las regiones de decisión (más claro) y puntos (más oscuro)
    # Regiones: 0 = Rojo claro, 1 = Azul claro, 2 = Verde claro
    cmap_light = ListedColormap(['#ffcccc', '#cce5ff', '#e2f0d9'])
    # Puntos: 0 = Rojo oscuro, 1 = Azul oscuro, 2 = Verde oscuro
    colors_dark = ['#cc0000', '#0066cc', '#385723']

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # Asegurar que results/ exista
    results_dir = RESULTS_PATH
    results_dir.mkdir(parents=True, exist_ok=True)

    for col_idx, seed in enumerate(seeds):
        # Cargar datos
        data_path = DATASETS_PATH / f"spiral_splits_seed_{seed}.npz"
        if not data_path.exists():
            print(f"Error: No se encontró {data_path}")
            return
        
        data = np.load(data_path)
        X_retain, y_retain = data["X_retain"], data["y_retain"]
        X_forget, y_forget = data["X_forget"], data["y_forget"]
        
        X_all = np.concatenate([X_retain, X_forget], axis=0)
        y_all = np.concatenate([y_retain, y_forget], axis=0)
        
        num_features = X_retain.shape[1]
        num_classes = int(y_all.max() + 1)

        # Rango para la cuadrícula
        x_min, x_max = X_all[:, 0].min() - 0.5, X_all[:, 0].max() + 0.5
        y_min, y_max = X_all[:, 1].min() - 0.5, X_all[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        grid_points_t = torch.tensor(grid_points, dtype=torch.float32).to(device)

        for row_idx, (model_name, model_label) in enumerate(model_types):
            ax = axes[row_idx, col_idx]

            # Instanciar el modelo
            model = model_class(input_dim=num_features, hidden_dim=hidden_dim, output_dim=num_classes)
            model = model.to(device)

            # Cargar pesos
            weights_path = MODELS_PATH / "weights" / f"{model_name}_model_seed_{seed}.pth"
            if not weights_path.exists():
                print(f"Advertencia: No se encontraron los pesos en {weights_path}")
                ax.text(0.5, 0.5, f"Pesos no encontrados\n{model_name}", ha='center', va='center')
                continue

            model.load_state_dict(torch.load(weights_path, map_location=device))
            model.eval()

            # Predicción en la cuadrícula
            with torch.no_grad():
                outputs = model(grid_points_t)
                preds = outputs.argmax(dim=1).cpu().numpy()
            
            preds_grid = preds.reshape(xx.shape)

            # Graficar regiones de decisión
            ax.pcolormesh(xx, yy, preds_grid, cmap=cmap_light, shading='auto', alpha=0.5)

            # Graficar puntos de Retain (círculos coloreados por clase)
            ax.scatter(X_retain[:, 0], X_retain[:, 1], c=[colors_dark[int(c)] for c in y_retain],
                       s=20, alpha=0.8, edgecolors='none', label='Retain')

            # Graficar puntos de Forget (cruces negras)
            ax.scatter(X_forget[:, 0], X_forget[:, 1], c='black', marker='X',
                       s=45, alpha=0.9, label='Forget')

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks([])
            ax.set_yticks([])

            # Establecer títulos
            if row_idx == 0:
                ax.set_title(f"Seed {seed}\n{model_label}", fontsize=12, fontweight='bold')
            else:
                ax.set_title(model_label, fontsize=12, fontweight='bold')

            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='upper right')

    plt.tight_layout()
    
    # Determinar ruta de guardado
    if args.output_path:
        save_path = Path(args.output_path)
    else:
        save_path = results_dir / f"decision_boundaries_{args.unlearned_name}.png"

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Imagen de fronteras de decisión guardada exitosamente en: {save_path.resolve()}")

if __name__ == "__main__":
    main()
