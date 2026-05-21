"""
run_pipeline.py
───────────────────────────────────────────────
Script de automatización para ejecutar todo el pipeline del benchmark
con el dataset CIFAR-nano y la arquitectura ResNet-18.
"""

import sys
import subprocess
import time
from pathlib import Path


def run_command(cmd, desc):
    print(f"\n========================================================")
    print(f" EJECUTANDO: {desc}")
    print(f" Comando: {' '.join(cmd)}")
    print(f"========================================================\n")
    
    start_time = time.perf_counter()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.perf_counter() - start_time
    
    if result.returncode != 0:
        print(f"\n[ERROR] El comando falló con código {result.returncode} tras {elapsed:.2f}s.")
        sys.exit(result.returncode)
        
    print(f"\n[ÉXITO] Completado en {elapsed:.2f}s.\n")


def main():
    print("Iniciando pipeline completo del Benchmark con CIFAR-nano y ResNet-18...\n")
    
    # Directorio base
    cwd = str(Path(__file__).parent.resolve())
    
    # Python executable
    py = sys.executable

    # 1. Búsqueda de hiperparámetros estándar (Optuna)
    # Usamos 5 trials para que la demostración sea rápida
    run_command(
        [py, "00_hp_search.py", "--protocol", "standard", "--dataset", "cifar_nano", 
         "--model_arch", "models.resnet.ResNet18", "--n_trials", "5", "--seed", "0"],
        "Paso 1: Búsqueda de HP estándar con Optuna (5 trials)"
    )

    # 2. Entrenamiento del modelo base (con retain + forget) para seeds 0, 1, 2
    run_command(
        [py, "3_train_model.py", "--model_arch", "models.resnet.ResNet18", "--protocol", "standard",
         "--train_splits", "retain,forget", "--model_name", "base", "--dataset", "cifar_nano", "--seeds", "0,1,2"],
        "Paso 2: Entrenamiento del modelo Base (Retain + Forget) para seeds 0, 1, 2"
    )

    # 3. Entrenamiento del modelo naive (solo retain) para seeds 0, 1, 2
    run_command(
        [py, "3_train_model.py", "--model_arch", "models.resnet.ResNet18", "--protocol", "standard",
         "--train_splits", "retain", "--model_name", "naive", "--dataset", "cifar_nano", "--seeds", "0,1,2"],
        "Paso 3: Entrenamiento del modelo Naive (solo Retain) para seeds 0, 1, 2"
    )

    # 4. Búsqueda de hiperparámetros para desaprendizaje CFK
    run_command(
        [py, "00_hp_search.py", "--protocol", "cfk", "--dataset", "cifar_nano", 
         "--model_arch", "models.resnet.ResNet18", "--n_trials", "5", "--seed", "0"],
        "Paso 4: Búsqueda de HP para desaprendizaje CFK (5 trials)"
    )

    # 5. Aplicar desaprendizaje CFK para seeds 0, 1, 2
    run_command(
        [py, "3_train_model.py", "--model_arch", "models.resnet.ResNet18", "--protocol", "cfk",
         "--train_splits", "retain", "--model_name", "cfk", "--dataset", "cifar_nano", 
         "--pretrained_weights", "models/weights/base_model_seed_{seed}.pth", 
         "--hp_file", "models/best_cfk_hp.json", "--seeds", "0,1,2"],
        "Paso 5: Aplicación del protocolo CFK para seeds 0, 1, 2"
    )

    # 6. Búsqueda de hiperparámetros para desaprendizaje EUK
    run_command(
        [py, "00_hp_search.py", "--protocol", "euk", "--dataset", "cifar_nano", 
         "--model_arch", "models.resnet.ResNet18", "--n_trials", "5", "--seed", "0"],
        "Paso 6: Búsqueda de HP para desaprendizaje EUK (5 trials)"
    )

    # 7. Aplicar desaprendizaje EUK para seeds 0, 1, 2
    run_command(
        [py, "3_train_model.py", "--model_arch", "models.resnet.ResNet18", "--protocol", "euk",
         "--train_splits", "retain", "--model_name", "euk", "--dataset", "cifar_nano", 
         "--pretrained_weights", "models/weights/base_model_seed_{seed}.pth", 
         "--hp_file", "models/best_euk_hp.json", "--seeds", "0,1,2"],
        "Paso 7: Aplicación del protocolo EUK para seeds 0, 1, 2"
    )

    # 8. Evaluación de métricas para CFK
    run_command(
        [py, "4_metricas.py", "--unlearned_name", "cfk", "--model_arch", "models.resnet.ResNet18", 
         "--dataset", "cifar_nano", "--seeds", "0,1,2"],
        "Paso 8: Evaluación de métricas para el protocolo CFK"
    )

    # 9. Evaluación de métricas para EUK
    run_command(
        [py, "4_metricas.py", "--unlearned_name", "euk", "--model_arch", "models.resnet.ResNet18", 
         "--dataset", "cifar_nano", "--seeds", "0,1,2"],
        "Paso 9: Evaluación de métricas para el protocolo EUK"
    )

    print("\n========================================================")
    print(" ¡PIPELINE COMPLETADO CON ÉXITO! ")
    print("========================================================\n")


if __name__ == "__main__":
    main()
