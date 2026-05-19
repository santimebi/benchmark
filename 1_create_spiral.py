"""
1_create_spiral.py
───────────────────────────────────────────────
Paso 1 del pipeline: Generación del dataset espiral.

Genera un dataset sintético de espirales entrelazadas con N clases,
donde cada clase describe una espiral con ruido controlado.
El resultado se guarda como CSV en ``DATASETS_PATH/spiral.csv``.

Uso:
    python 1_create_spiral.py
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.config import DATASETS_PATH


def make_spiral_dataset(
    file_name=DATASETS_PATH / "spiral.csv",
    overwrite_file=False,
    n_points_per_class=400,
    n_classes=3,
    noise=0.45,
    rotations=2.5,
    random_state=42,
    verbose=True,
    plot_dataset=False,
):
    """
    Genera un dataset sintético de espirales entrelazadas.

    Cada clase genera ``n_points_per_class`` puntos a lo largo de una
    espiral con desfase angular uniforme. El ruido gaussiano se aplica
    sobre el ángulo para dificultar la separación.

    Args:
        file_name (Path): Ruta de salida del CSV.
        overwrite_file (bool): Si es ``True``, sobreescribe el CSV existente.
        n_points_per_class (int): Puntos por clase.
        n_classes (int): Número de espirales / clases.
        noise (float): Desviación estándar del ruido angular.
        rotations (float): Número de rotaciones completas de cada espiral.
        random_state (int): Semilla para reproducibilidad.
        verbose (bool): Imprime mensajes de estado.
        plot_dataset (bool): Muestra una visualización matplotlib.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(X, y)`` donde ``X`` tiene forma
            ``(n_points_per_class * n_classes, 2)`` e ``y`` las etiquetas.
    """
    np.random.seed(random_state)

    X = []
    y = []

    for class_id in range(n_classes):
        # Radio creciente
        r = np.linspace(0.0, 1.0, n_points_per_class)

        # Ángulo de la espiral
        theta = np.linspace(0, rotations * 2 * np.pi, n_points_per_class)

        # Desfase para separar las espirales
        offset = class_id * 2 * np.pi / n_classes

        # Añadimos ruido
        theta_noise = theta + np.random.normal(0, noise, n_points_per_class)

        x1 = r * np.cos(theta_noise + offset)
        x2 = r * np.sin(theta_noise + offset)

        X_class = np.stack([x1, x2], axis=1)
        y_class = np.full(n_points_per_class, class_id)

        X.append(X_class)
        y.append(y_class)

    X = np.vstack(X)
    y = np.concatenate(y)

    file_path = file_name
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if not file_path.exists() or overwrite_file:
        data = np.column_stack((X, y))
        np.savetxt(file_path, data, delimiter=",", header="x1,x2,label", comments="")
        if verbose:
            print(f"Dataset guardado en {file_path}")
    else:
        if verbose:
            print(f"El archivo {file_path} ya existe. Saltando guardado (overwrite_file=False).")

    if plot_dataset:
        print("Graficando dataset...")
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.show()
        plt.waitforbuttonpress()
        plt.close()
        
    return X, y
    

if __name__ == "__main__":

    make_spiral_dataset()

