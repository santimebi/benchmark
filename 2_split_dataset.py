"""
2_split_dataset.py
───────────────────────────────────────────────
Paso 2 del pipeline: Partición del dataset espiral.

Divide el dataset en cuatro subconjuntos:
  - **Retain**: datos de entrenamiento que el modelo debe recordar.
  - **Forget**: datos (exclusivamente clase 0) que el modelo debe olvidar.
  - **Validation**: datos para evaluar durante el entrenamiento.
  - **Test**: datos reservados para la evaluación final.

El forget set se construye seleccionando las muestras de la clase 0
más cercanas al centro de la espiral (radio menor), lo que simula
un escenario realista de machine unlearning.

Uso:
    python 2_split_dataset.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils.config import DATASETS_PATH


def split_dataset(
            file_name=DATASETS_PATH / "spiral.csv",
            verbose=True,
            plot_dataset=True,
            test_size=0.2,
            forget_size=0.2,
            random_state=42,
            val_size=0.1,
            forget_class=0,
            ):
    """
    Divide el dataset espiral en retain, forget, validation y test.

    La estrategia de forget aísla muestras de la clase 0 ordenadas por
    proximidad al origen (centro de la espiral), asegurando que el forget
    set sea un subconjunto coherente y espacialmente localizado.

    Args:
        file_name (Path): Ruta al CSV generado por ``1_create_spiral.py``.
        verbose (bool): Imprime estadísticas de los splits.
        plot_dataset (bool): Muestra gráfico retain vs forget.
        test_size (float): Proporción para el set de test.
        forget_size (float): Proporción del train dedicada a forget.
        random_state (int): Semilla para reproducibilidad.
        val_size (float): Proporción del train_val dedicada a validación.

    Returns:
        tuple: ``(X_retain, X_forget, X_val, X_test,
                  y_retain, y_forget, y_val, y_test)``
    """
    # Cargamos el archivo. Tiene columnas x1, x2, label
    data = np.loadtxt(file_name, delimiter=",", skiprows=1)
    
    # Separamos features (X) de target (y)
    X = data[:, :-1]
    y = data[:, -1]

    # Primera división: train_val, test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    
    # Segunda división: del train_val, sacamos el porcentaje para validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_size,
        random_state=random_state,
    )

    # Tercera división: del train, sacamos el tamaño de forget (ej. 20%)
    f_size = int(forget_size * len(X_train))
    
    # Indices de la clase elegida en X_train
    class_0_indices = np.where(y_train == forget_class)[0]
    
    # Distancia al centro (origen) para ordenar (r=0 es el centro de la espiral, que corresponde a theta=0)
    distances = np.linalg.norm(X_train[class_0_indices], axis=1)
    sorted_class_0_idx = class_0_indices[np.argsort(distances)]
    
    # Tomamos los primeros 'f_size' elementos de la clase elegida desde el centro
    forget_indices = sorted_class_0_idx[:f_size]
    
    # El resto de train pasa a ser retain
    retain_indices = np.setdiff1d(np.arange(len(X_train)), forget_indices)
    
    X_forget = X_train[forget_indices]
    y_forget = y_train[forget_indices]
    
    X_retain = X_train[retain_indices]
    y_retain = y_train[retain_indices]

    if verbose:
        total_len = len(data)
        print(f"Retain size: {len(X_retain)} ({(len(X_retain)/total_len)*100:.1f}%)")
        print(f"Forget size: {len(X_forget)} ({(len(X_forget)/total_len)*100:.1f}%)")
        print(f"Validation size: {len(X_val)} ({(len(X_val)/total_len)*100:.1f}%)")
        print(f"Test size: {len(X_test)} ({(len(X_test)/total_len)*100:.1f}%)")
        print(f"Verificación de Forget - Cantidad: {len(y_forget)} | Clases únicas: {np.unique(y_forget)}")

    if plot_dataset:
        plt.figure(figsize=(8, 6))
        # Retain en sus colores correspondientes
        plt.scatter(X_retain[:, 0], X_retain[:, 1], c=y_retain, cmap='viridis', label='Retain', alpha=0.6)
        # Forget exclusivamente en color negro
        plt.scatter(X_forget[:, 0], X_forget[:, 1], c='black', label='Forget', alpha=1.0)
        plt.title(f"Retain vs Forget Split (Seed: {random_state})")
        
        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, label=f'Forget (Clase {forget_class})'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#440154', markersize=10, label='Retain (Clase 0)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#21918c', markersize=10, label='Retain (Clase 1)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#fde725', markersize=10, label='Retain (Clase 2)')
        ]
        plt.legend(handles=custom_lines)
        plt.show()
        plt.close()

    return X_retain, X_forget, X_val, X_test, y_retain, y_forget, y_val, y_test


def save_all_splits(file_name, output_dir, seeds=[0, 1, 2], overwrite_file=False, verbose=True, plot_dataset=False, forget_class=0, forget_size=0.2):
    """
    Genera y guarda los splits del dataset en archivos ``.npz`` para múltiples seeds.

    Cada seed produce un fichero ``spiral_c{forget_class}_s{forget_size}_splits_seed_{seed}.npz`` que
    contiene las 8 arrays del split (X/y para retain, forget, val, test).

    Args:
        file_name (Path): Ruta al CSV del dataset espiral.
        output_dir (Path): Directorio de salida para los ``.npz``.
        seeds (list[int]): Lista de semillas a generar.
        overwrite_file (bool): Si es ``True``, regenera splits existentes.
        verbose (bool): Imprime mensajes de estado.
        plot_dataset (bool): Muestra gráfico por cada seed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for seed in seeds:
        output_file = output_dir / f"spiral_c{forget_class}_s{forget_size}_splits_seed_{seed}.npz"
        
        if not output_file.exists() or overwrite_file:
            if verbose:
                print(f"\n--- Generando splits para la seed {seed} ---")
            X_retain, X_forget, X_val, X_test, y_retain, y_forget, y_val, y_test = split_dataset(
                file_name=file_name,
                random_state=seed,
                verbose=verbose,
                plot_dataset=plot_dataset,
                forget_size=forget_size,
                forget_class=forget_class,
            )
            
            np.savez(
                output_file,
                X_retain=X_retain, y_retain=y_retain,
                X_forget=X_forget, y_forget=y_forget,
                X_val=X_val, y_val=y_val,
                X_test=X_test, y_test=y_test
            )
            if verbose:
                print(f"Guardado exitosamente en: {output_file}")
        else:
            if verbose:
                print(f"\n--- El archivo {output_file.name} ya existe. Saltando generación (overwrite_file=False) ---")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Particionar dataset de espiral con configuración parametrizable.")
    parser.add_argument("--forget_class", type=int, default=0, help="Clase a olvidar (default: 0)")
    parser.add_argument("--forget_size", type=float, default=0.2, help="Proporción del train dedicada a forget (default: 0.2)")
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Semillas separadas por comas (default: 0,1,2)")
    parser.add_argument("--overwrite", action="store_true", help="Sobrescribe archivos existentes")
    args = parser.parse_args()
    
    seeds_list = [int(s.strip()) for s in args.seeds.split(",")]
    
    save_all_splits(
        file_name=DATASETS_PATH / "spiral.csv",
        output_dir=DATASETS_PATH,
        seeds=seeds_list,
        overwrite_file=args.overwrite,
        verbose=True,
        plot_dataset=False,
        forget_class=args.forget_class,
        forget_size=args.forget_size
    )
