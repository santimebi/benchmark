import subprocess
import sys
import argparse
from pathlib import Path
from utils.config import MODELS_PATH

def main():
    parser = argparse.ArgumentParser(description="Ejecutar experimentos de desaprendizaje para distintos valores de k.")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        help="Nombre del dataset a utilizar (ej. cifar10, cifar_nano, spiral)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Número de épocas de ajuste para el desaprendizaje (default: 20)")
    parser.add_argument("--seeds", type=str, default="0,1,2",
                        help="Semillas separadas por comas (default: 0,1,2)")
    parser.add_argument("--model_arch", type=str, default="models.resnet.ResNet18",
                        help="Arquitectura del modelo (default: models.resnet.ResNet18)")
    parser.add_argument("--k_values", type=str, default="17,13,9,5",
                        help="Valores de k separados por comas (default: 17,13,9,5)")
    args = parser.parse_args()

    protocols = ["cfk", "euk", "cfgk"]
    k_values = [int(k.strip()) for k in args.k_values.split(",")]
    epochs = args.epochs
    seeds = args.seeds
    dataset = args.dataset
    model_arch = args.model_arch
    
    python_exec = sys.executable

    for protocol in protocols:
        for k in k_values:
            model_name = f"{protocol}_k{k}"
            print(f"\n========================================================")
            
            # Buscar el archivo de mejores hiperparámetros específico del protocolo
            hp_file = MODELS_PATH / f"best_{protocol}_hp.json"
            if not hp_file.exists():
                hp_file = MODELS_PATH / "best_hp.json"
                
            cmd_train = [
                python_exec, "3_train_model.py",
                "--model_arch", model_arch,
                "--protocol", protocol,
                "--train_splits", "retain",
                "--model_name", model_name,
                "--dataset", dataset,
                "--pretrained_weights", str(MODELS_PATH / "weights/base_model_seed_{seed}.pth"),
                "--hp_file", str(hp_file),
                "--epochs", str(epochs),
                "--k", str(k),
                "--seeds", seeds
            ]
            print(f"EJECUTANDO ENTRENAMIENTO: {protocol} con k={k}")
            print(f"Comando: {' '.join(cmd_train)}")
            subprocess.run(cmd_train, check=True)
            
            # Ejecutar métricas automáticamente para este experimento
            cmd_metrics = [
                python_exec, "4_metricas.py",
                "--unlearned_name", model_name,
                "--model_arch", model_arch,
                "--dataset", dataset,
                "--seeds", seeds,
                "--hp_file", str(MODELS_PATH / "best_hp.json")
            ]
            print(f"EJECUTANDO EVALUACIÓN DE MÉTRICAS: {model_name}")
            print(f"Comando: {' '.join(cmd_metrics)}")
            subprocess.run(cmd_metrics, check=True)

    print("\n========================================================")
    print(" ¡TODAS LAS PRUEBAS DE K HAN SIDO COMPLETADAS CON ÉXITO! ")
    print("========================================================\n")

if __name__ == "__main__":
    main()
