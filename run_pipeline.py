import argparse
import sys
import subprocess
import time
from pathlib import Path
from utils.config import MODELS_PATH


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
    parser.add_argument("--dataset", type=str, default="cifar_nano", choices=["cifar_nano", "spiral", "cifar10"],
                        help="Nombre del dataset a utilizar (cifar_nano, spiral, o cifar10)")
    parser.add_argument("--model_arch", type=str, default="models.resnet.ResNet18",
                        help="Import path de la arquitectura del modelo")
    parser.add_argument("--n_trials", type=int, default=5,
                        help="Número de trials de optimización Optuna")
    parser.add_argument("--seeds", type=str, default="0,1,2",
                        help="Semillas separadas por comas")
    parser.add_argument("--rk_tau", type=float, default=0.03,
                        help="Escala de perturbación para Residual Knowledge (default: 0.03)")
    parser.add_argument("--rk_c", type=int, default=100,
                        help="Número de perturbaciones Monte Carlo para Residual Knowledge (default: 100)")
    parser.add_argument("--rk_chunk_size", type=int, default=100,
                        help="Tamaño de lote para evaluaciones vectorizadas de RK (default: 100)")
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
         "--pretrained_weights", str(MODELS_PATH / "weights/base_model_seed_{seed}.pth"), 
         "--hp_file", str(MODELS_PATH / "best_cfk_hp.json"), "--seeds", args.seeds],
        "Paso 5: Aplicación del protocolo CFK"
    )

    # 5.1. Búsqueda de hiperparámetros para desaprendizaje CFGK
    run_command(
        [py, "00_hp_search.py", "--protocol", "cfgk", "--dataset", args.dataset, 
         "--model_arch", args.model_arch, "--n_trials", str(args.n_trials), "--seed", "0"],
        "Paso 5.1: Búsqueda de HP para desaprendizaje CFGK"
    )

    # 5.2. Aplicar desaprendizaje CFGK para las semillas
    run_command(
        [py, "3_train_model.py", "--model_arch", args.model_arch, "--protocol", "cfgk",
         "--train_splits", "retain", "--model_name", "cfgk", "--dataset", args.dataset, 
         "--pretrained_weights", str(MODELS_PATH / "weights/base_model_seed_{seed}.pth"), 
         "--hp_file", str(MODELS_PATH / "best_cfgk_hp.json"), "--seeds", args.seeds],
        "Paso 5.2: Aplicación del protocolo CFGK"
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
         "--pretrained_weights", str(MODELS_PATH / "weights/base_model_seed_{seed}.pth"), 
         "--hp_file", str(MODELS_PATH / "best_euk_hp.json"), "--seeds", args.seeds],
        "Paso 7: Aplicación del protocolo EUK"
    )

    # 7.1. Búsqueda de hiperparámetros para desaprendizaje RURK (Optuna con 1 trial)
    run_command(
        [py, "00_hp_search.py", "--protocol", "rurk", "--dataset", args.dataset, 
         "--model_arch", args.model_arch, "--n_trials", "1", "--seed", "0"],
        "Paso 7.1: Búsqueda de HP para desaprendizaje RURK"
    )

    # 7.2. Aplicar desaprendizaje RURK para las semillas
    run_command(
        [py, "3_train_model.py", "--model_arch", args.model_arch, "--protocol", "rurk",
         "--train_splits", "retain", "--model_name", "rurk", "--dataset", args.dataset, 
         "--pretrained_weights", str(MODELS_PATH / "weights/base_model_seed_{seed}.pth"), 
         "--hp_file", str(MODELS_PATH / "best_rurk_hp.json"), "--seeds", args.seeds],
        "Paso 7.2: Aplicación del protocolo RURK"
    )

    # 8. Evaluación de métricas para CFK
    run_command(
        [py, "4_metricas.py", "--unlearned_name", "cfk", "--model_arch", args.model_arch, 
         "--dataset", args.dataset, "--seeds", args.seeds,
         "--rk_tau", str(args.rk_tau), "--rk_c", str(args.rk_c), "--rk_chunk_size", str(args.rk_chunk_size)],
        "Paso 8: Evaluación de métricas para el protocolo CFK"
    )

    # 8.5. Evaluación de métricas para CFGK
    run_command(
        [py, "4_metricas.py", "--unlearned_name", "cfgk", "--model_arch", args.model_arch, 
         "--dataset", args.dataset, "--seeds", args.seeds,
         "--rk_tau", str(args.rk_tau), "--rk_c", str(args.rk_c), "--rk_chunk_size", str(args.rk_chunk_size)],
        "Paso 8.5: Evaluación de métricas para el protocolo CFGK"
    )

    # 9. Evaluación de métricas para EUK
    run_command(
        [py, "4_metricas.py", "--unlearned_name", "euk", "--model_arch", args.model_arch, 
         "--dataset", args.dataset, "--seeds", args.seeds,
         "--rk_tau", str(args.rk_tau), "--rk_c", str(args.rk_c), "--rk_chunk_size", str(args.rk_chunk_size)],
        "Paso 9: Evaluación de métricas para el protocolo EUK"
    )

    # 9.5. Evaluación de métricas para RURK
    run_command(
        [py, "4_metricas.py", "--unlearned_name", "rurk", "--model_arch", args.model_arch, 
         "--dataset", args.dataset, "--seeds", args.seeds,
         "--rk_tau", str(args.rk_tau), "--rk_c", str(args.rk_c), "--rk_chunk_size", str(args.rk_chunk_size)],
        "Paso 9.5: Evaluación de métricas para el protocolo RURK"
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
