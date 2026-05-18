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
    plot_dataset=True,
):
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

