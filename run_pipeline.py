import argparse
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
    parser = argparse.ArgumentParser(description="Script de automatización para ejecutar todo el pipeline del benchmark.")
    parser.add_argument("--dataset", type=str, default="cifar_nano", choices=["cifar_nano", "spiral"],
                        help="Nombre del dataset a utilizar (cifar_nano o spiral)")
    parser.add_argument("--model_arch", type=str, default="models.resnet.ResNet18",
                        help="Import path de la arquitectura del modelo")
    parser.add_argument("--n_trials", type=int, default=5,
                        help="Número de trials de optimización Optuna")
    parser.add_argument("--seeds", type=str, default="0,1,2",
                        help="Semillas separadas por comas")
    args = parser.parse_args()

    print(f"Iniciando pipeline completo del Benchmark con {args.dataset} y {args.model_arch}...\n")
    
    # Directorio base
    cwd = str(Path(__file__).parent.resolve())
    
    # Python executable
    py = sys.executable

    # 1. Búsqueda de hiperparámetros estándar (Optuna)
    run_command(
        [py, "00_hp_search.py", "--protocol", "standard", "--dataset", args.dataset, 
         "--model_arch", args.model_arch, "--n_trials", str(args.n_trials), "--seed", "0"],
        "Paso 1: Búsqueda de HP estándar con Optuna"
    )

    # 2. Entrenamiento del modelo base (con retain + forget) para las semillas
    run_command(
        [py, "3_train_model.py", "--model_arch", args.model_arch, "--protocol", "standard",
         "--train_splits", "retain,forget", "--model_name", "base", "--dataset", args.dataset, "--seeds", args.seeds],
        "Paso 2: Entrenamiento del modelo Base (Retain + Forget)"
    )

    # 3. Entrenamiento del modelo naive (solo retain) para las semillas
    run_command(
        [py, "3_train_model.py", "--model_arch", args.model_arch, "--protocol", "standard",
         "--train_splits", "retain", "--model_name", "naive", "--dataset", args.dataset, "--seeds", args.seeds],
        "Paso 3: Entrenamiento del modelo Naive (solo Retain)"
    )

    # 4. Búsqueda de hiperparámetros para desaprendizaje CFK
    run_command(
        [py, "00_hp_search.py", "--protocol", "cfk", "--dataset", args.dataset, 
         "--model_arch", args.model_arch, "--n_trials", str(args.n_trials), "--seed", "0"],
        "Paso 4: Búsqueda de HP para desaprendizaje CFK"
    )

    # 5. Aplicar desaprendizaje CFK para las semillas
    run_command(
        [py, "3_train_model.py", "--model_arch", args.model_arch, "--protocol", "cfk",
         "--train_splits", "retain", "--model_name", "cfk", "--dataset", args.dataset, 
         "--pretrained_weights", "models/weights/base_model_seed_{seed}.pth", 
         "--hp_file", "models/best_cfk_hp.json", "--seeds", args.seeds],
        "Paso 5: Aplicación del protocolo CFK"
    )

    # 6. Búsqueda de hiperparámetros para desaprendizaje EUK
    run_command(
        [py, "00_hp_search.py", "--protocol", "euk", "--dataset", args.dataset, 
         "--model_arch", args.model_arch, "--n_trials", str(args.n_trials), "--seed", "0"],
        "Paso 6: Búsqueda de HP para desaprendizaje EUK"
    )

    # 7. Aplicar desaprendizaje EUK para las semillas
    run_command(
        [py, "3_train_model.py", "--model_arch", args.model_arch, "--protocol", "euk",
         "--train_splits", "retain", "--model_name", "euk", "--dataset", args.dataset, 
         "--pretrained_weights", "models/weights/base_model_seed_{seed}.pth", 
         "--hp_file", "models/best_euk_hp.json", "--seeds", args.seeds],
        "Paso 7: Aplicación del protocolo EUK"
    )

    # 8. Evaluación de métricas para CFK
    run_command(
        [py, "4_metricas.py", "--unlearned_name", "cfk", "--model_arch", args.model_arch, 
         "--dataset", args.dataset, "--seeds", args.seeds],
        "Paso 8: Evaluación de métricas para el protocolo CFK"
    )

    # 9. Evaluación de métricas para EUK
    run_command(
        [py, "4_metricas.py", "--unlearned_name", "euk", "--model_arch", args.model_arch, 
         "--dataset", args.dataset, "--seeds", args.seeds],
        "Paso 9: Evaluación de métricas para el protocolo EUK"
    )

    # 10. Generar tabla de métricas resumida
    run_command(
        [py, "5_generate_tables.py"],
        "Paso 10: Generación y guardado de tabla de métricas"
    )

    print("\n========================================================")
    print(" ¡PIPELINE COMPLETADO CON ÉXITO! ")
    print("========================================================\n")


if __name__ == "__main__":
    main()
